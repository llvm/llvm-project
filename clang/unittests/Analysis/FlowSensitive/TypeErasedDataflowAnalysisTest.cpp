//===- unittests/Analysis/FlowSensitive/TypeErasedDataflowAnalysisTest.cpp ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/DebugSupport.h"
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/ADT/StringMapEntry.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace clang;
using namespace dataflow;
using namespace test;
using namespace ast_matchers;
using llvm::IsStringMapEntry;
using ::testing::DescribeMatcher;
using ::testing::IsEmpty;
using ::testing::NotNull;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

class DataflowAnalysisTest : public Test {
protected:
  template <typename AnalysisT>
  llvm::Expected<std::vector<
      std::optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>>
  runAnalysis(llvm::StringRef Code, AnalysisT (*MakeAnalysis)(ASTContext &)) {
    AST = tooling::buildASTFromCodeWithArgs(Code, {"-std=c++11"});

    auto *Func = selectFirst<FunctionDecl>(
        "func",
        match(functionDecl(ast_matchers::hasName("target")).bind("func"),
              AST->getASTContext()));
    assert(Func != nullptr);

    ACFG =
        std::make_unique<AdornedCFG>(llvm::cantFail(AdornedCFG::build(*Func)));

    AnalysisT Analysis = MakeAnalysis(AST->getASTContext());
    DACtx = std::make_unique<DataflowAnalysisContext>(
        std::make_unique<WatchedLiteralsSolver>());
    Environment Env(*DACtx, *Func);

    return runDataflowAnalysis(*ACFG, Analysis, Env);
  }

  /// Returns the `CFGBlock` containing `S` (and asserts that it exists).
  const CFGBlock *blockForStmt(const Stmt &S) {
    const CFGBlock *Block = ACFG->getStmtToBlock().lookup(&S);
    assert(Block != nullptr);
    return Block;
  }

  template <typename StateT>
  const StateT &
  blockStateForStmt(const std::vector<std::optional<StateT>> &BlockStates,
                    const Stmt &S) {
    const std::optional<StateT> &MaybeState =
        BlockStates[blockForStmt(S)->getBlockID()];
    assert(MaybeState.has_value());
    return *MaybeState;
  }

  /// Returns the first node that matches `Matcher` (and asserts that the match
  /// was successful, i.e. the returned node is not null).
  template <typename NodeT, typename MatcherT>
  const NodeT &matchNode(MatcherT Matcher) {
    const auto *Node = selectFirst<NodeT>(
        "node", match(Matcher.bind("node"), AST->getASTContext()));
    assert(Node != nullptr);
    return *Node;
  }

  std::unique_ptr<ASTUnit> AST;
  std::unique_ptr<AdornedCFG> ACFG;
  std::unique_ptr<DataflowAnalysisContext> DACtx;
};

TEST_F(DataflowAnalysisTest, NoopAnalysis) {
  auto BlockStates = llvm::cantFail(
      runAnalysis<NoopAnalysis>("void target() {}", [](ASTContext &C) {
        return NoopAnalysis(C,
                            // Don't use builtin transfer function.
                            DataflowAnalysisOptions{std::nullopt});
      }));
  EXPECT_EQ(BlockStates.size(), 2u);
  EXPECT_TRUE(BlockStates[0].has_value());
  EXPECT_TRUE(BlockStates[1].has_value());
}

// Basic test that `diagnoseFunction` calls the Diagnoser function for the
// number of elements expected.
TEST_F(DataflowAnalysisTest, DiagnoseFunctionDiagnoserCalledOnEachElement) {
  std::string Code = R"(void target() { int x = 0; ++x; })";
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-std=c++11"});

  auto *Func =
      cast<FunctionDecl>(findValueDecl(AST->getASTContext(), "target"));
  auto Diagnoser = [](const CFGElement &Elt, ASTContext &,
                      const TransferStateForDiagnostics<NoopLattice> &) {
    llvm::SmallVector<std::string> Diagnostics(1);
    llvm::raw_string_ostream OS(Diagnostics.front());
    Elt.dumpToStream(OS);
    return Diagnostics;
  };
  auto Result = diagnoseFunction<NoopAnalysis, std::string>(
      *Func, AST->getASTContext(), Diagnoser);
  // `diagnoseFunction` provides no guarantees about the order in which elements
  // are visited, so we use `UnorderedElementsAre`.
  EXPECT_THAT_EXPECTED(Result, llvm::HasValue(UnorderedElementsAre(
                                   "0\n", "int x = 0;\n", "x\n", "++x\n",
                                   " (Lifetime ends)\n")));
}

// Tests for the statement-to-block map.
using StmtToBlockTest = DataflowAnalysisTest;

TEST_F(StmtToBlockTest, ConditionalOperator) {
  std::string Code = R"(
    void target(bool b) {
      int i = b ? 1 : 0;
    }
  )";
  ASSERT_THAT_ERROR(runAnalysis<NoopAnalysis>(
                        Code, [](ASTContext &C) { return NoopAnalysis(C); })
                        .takeError(),
                    llvm::Succeeded());

  const auto &IDecl = matchNode<DeclStmt>(declStmt(has(varDecl(hasName("i")))));
  const auto &ConditionalOp =
      matchNode<ConditionalOperator>(conditionalOperator());

  // The conditional operator should be associated with the same block as the
  // `DeclStmt` for `i`. (Specifically, the conditional operator should not be
  // associated with the block for which it is the terminator.)
  EXPECT_EQ(blockForStmt(IDecl), blockForStmt(ConditionalOp));
}

TEST_F(StmtToBlockTest, LogicalAnd) {
  std::string Code = R"(
    void target(bool b1, bool b2) {
      bool b = b1 && b2;
    }
  )";
  ASSERT_THAT_ERROR(runAnalysis<NoopAnalysis>(
                        Code, [](ASTContext &C) { return NoopAnalysis(C); })
                        .takeError(),
                    llvm::Succeeded());

  const auto &BDecl = matchNode<DeclStmt>(declStmt(has(varDecl(hasName("b")))));
  const auto &AndOp =
      matchNode<BinaryOperator>(binaryOperator(hasOperatorName("&&")));

  // The `&&` operator should be associated with the same block as the
  // `DeclStmt` for `b`. (Specifically, the `&&` operator should not be
  // associated with the block for which it is the terminator.)
  EXPECT_EQ(blockForStmt(BDecl), blockForStmt(AndOp));
}

TEST_F(StmtToBlockTest, IfStatementWithLogicalAnd) {
  std::string Code = R"(
    void target(bool b1, bool b2) {
      if (b1 && b2)
        ;
    }
  )";
  ASSERT_THAT_ERROR(runAnalysis<NoopAnalysis>(
                        Code, [](ASTContext &C) { return NoopAnalysis(C); })
                        .takeError(),
                    llvm::Succeeded());

  const auto &If = matchNode<IfStmt>(ifStmt());
  const auto &B2 =
      matchNode<DeclRefExpr>(declRefExpr(to(varDecl(hasName("b2")))));
  const auto &AndOp =
      matchNode<BinaryOperator>(binaryOperator(hasOperatorName("&&")));

  // The if statement is the terminator for the block that contains both `b2`
  // and the `&&` operator (which appears only as a terminator condition, not
  // as a regular `CFGElement`).
  const CFGBlock *IfBlock = blockForStmt(If);
  const CFGBlock *B2Block = blockForStmt(B2);
  const CFGBlock *AndOpBlock = blockForStmt(AndOp);
  EXPECT_EQ(IfBlock, B2Block);
  EXPECT_EQ(IfBlock, AndOpBlock);
}

// Tests that check we discard state for expressions correctly.
using DiscardExprStateTest = DataflowAnalysisTest;

TEST_F(DiscardExprStateTest, WhileStatement) {
  std::string Code = R"(
    void foo(int *p);
    void target(int *p) {
      while (p != nullptr)
        foo(p);
    }
  )";
  auto BlockStates = llvm::cantFail(runAnalysis<NoopAnalysis>(
      Code, [](ASTContext &C) { return NoopAnalysis(C); }));

  const auto &NotEqOp =
      matchNode<BinaryOperator>(binaryOperator(hasOperatorName("!=")));
  const auto &CallFoo =
      matchNode<CallExpr>(callExpr(callee(functionDecl(hasName("foo")))));

  // In the block that evaluates the expression `p != nullptr`, this expression
  // is associated with a value.
  const auto &NotEqOpState = blockStateForStmt(BlockStates, NotEqOp);
  EXPECT_NE(NotEqOpState.Env.getValue(NotEqOp), nullptr);

  // In the block that calls `foo(p)`, the value for `p != nullptr` is discarded
  // because it is not consumed outside the block it is in.
  const auto &CallFooState = blockStateForStmt(BlockStates, CallFoo);
  EXPECT_EQ(CallFooState.Env.getValue(NotEqOp), nullptr);
}

TEST_F(DiscardExprStateTest, BooleanOperator) {
  std::string Code = R"(
    void f();
    void target(bool b1, bool b2) {
      if (b1 && b2)
        f();
    }
  )";
  auto BlockStates = llvm::cantFail(runAnalysis<NoopAnalysis>(
      Code, [](ASTContext &C) { return NoopAnalysis(C); }));

  const auto &AndOp =
      matchNode<BinaryOperator>(binaryOperator(hasOperatorName("&&")));
  const auto &CallF =
      matchNode<CallExpr>(callExpr(callee(functionDecl(hasName("f")))));

  // In the block that evaluates the LHS of the `&&` operator, the LHS is
  // associated with a value, while the right-hand side is not (unsurprisingly,
  // as it hasn't been evaluated yet).
  const auto &LHSState = blockStateForStmt(BlockStates, *AndOp.getLHS());
  auto *LHSValue = cast<BoolValue>(LHSState.Env.getValue(*AndOp.getLHS()));
  EXPECT_NE(LHSValue, nullptr);
  EXPECT_EQ(LHSState.Env.getValue(*AndOp.getRHS()), nullptr);

  // In the block that evaluates the RHS, both the LHS and RHS are associated
  // with values, as they are both subexpressions of the `&&` operator, which
  // is evaluated in a later block.
  const auto &RHSState = blockStateForStmt(BlockStates, *AndOp.getRHS());
  EXPECT_EQ(RHSState.Env.getValue(*AndOp.getLHS()), LHSValue);
  auto *RHSValue = RHSState.Env.get<BoolValue>(*AndOp.getRHS());
  EXPECT_NE(RHSValue, nullptr);

  // In the block that evaluates `b1 && b2`, the `&&` as well as its operands
  // are associated with values.
  const auto &AndOpState = blockStateForStmt(BlockStates, AndOp);
  EXPECT_EQ(AndOpState.Env.getValue(*AndOp.getLHS()), LHSValue);
  EXPECT_EQ(AndOpState.Env.getValue(*AndOp.getRHS()), RHSValue);
  // FIXME: this test is too strict. We want to check equivalence not equality;
  // as is, its a change detector test. Notice that we only evaluate `b1 && b2`
  // in a context where we know that `b1` is true, so there's a potential
  // optimization to store only `RHSValue` as the operation's value.
  EXPECT_EQ(AndOpState.Env.getValue(AndOp),
            &AndOpState.Env.makeAnd(*LHSValue, *RHSValue));

  // In the block that calls `f()`, none of `b1`, `b2`, or `b1 && b2` should be
  // associated with values.
  const auto &CallFState = blockStateForStmt(BlockStates, CallF);
  EXPECT_EQ(CallFState.Env.getValue(*AndOp.getLHS()), nullptr);
  EXPECT_EQ(CallFState.Env.getValue(*AndOp.getRHS()), nullptr);
  EXPECT_EQ(CallFState.Env.getValue(AndOp), nullptr);
}

TEST_F(DiscardExprStateTest, ConditionalOperator) {
  std::string Code = R"(
    void f(int*, int);
    void g();
    bool cond();

    void target() {
      int i = 0;
      if (cond())
        f(&i, cond() ? 1 : 0);
      g();
    }
  )";
  auto BlockStates = llvm::cantFail(runAnalysis<NoopAnalysis>(
      Code, [](ASTContext &C) { return NoopAnalysis(C); }));

  const auto &AddrOfI =
      matchNode<UnaryOperator>(unaryOperator(hasOperatorName("&")));
  const auto &CallF =
      matchNode<CallExpr>(callExpr(callee(functionDecl(hasName("f")))));
  const auto &CallG =
      matchNode<CallExpr>(callExpr(callee(functionDecl(hasName("g")))));

  // In the block that evaluates `&i`, it should obviously have a value.
  const auto &AddrOfIState = blockStateForStmt(BlockStates, AddrOfI);
  auto *AddrOfIVal = AddrOfIState.Env.get<PointerValue>(AddrOfI);
  EXPECT_NE(AddrOfIVal, nullptr);

  // Because of the conditional operator, the `f(...)` call is evaluated in a
  // different block than `&i`, but `&i` still needs to have a value here
  // because it's a subexpression of the call.
  const auto &CallFState = blockStateForStmt(BlockStates, CallF);
  EXPECT_NE(&CallFState, &AddrOfIState);
  EXPECT_EQ(CallFState.Env.get<PointerValue>(AddrOfI), AddrOfIVal);

  // In the block that calls `g()`, `&i` should no longer be associated with a
  // value.
  const auto &CallGState = blockStateForStmt(BlockStates, CallG);
  EXPECT_EQ(CallGState.Env.get<PointerValue>(AddrOfI), nullptr);
}

struct NonConvergingLattice {
  int State;

  bool operator==(const NonConvergingLattice &Other) const {
    return State == Other.State;
  }

  LatticeJoinEffect join(const NonConvergingLattice &Other) {
    if (Other.State == 0)
      return LatticeJoinEffect::Unchanged;
    State += Other.State;
    return LatticeJoinEffect::Changed;
  }
};

class NonConvergingAnalysis
    : public DataflowAnalysis<NonConvergingAnalysis, NonConvergingLattice> {
public:
  explicit NonConvergingAnalysis(ASTContext &Context)
      : DataflowAnalysis<NonConvergingAnalysis, NonConvergingLattice>(
            Context,
            // Don't apply builtin transfer function.
            DataflowAnalysisOptions{std::nullopt}) {}

  static NonConvergingLattice initialElement() { return {0}; }

  void transfer(const CFGElement &, NonConvergingLattice &E, Environment &) {
    ++E.State;
  }
};

TEST_F(DataflowAnalysisTest, NonConvergingAnalysis) {
  std::string Code = R"(
    void target() {
      while(true) {}
    }
  )";
  auto Res = runAnalysis<NonConvergingAnalysis>(
      Code, [](ASTContext &C) { return NonConvergingAnalysis(C); });
  EXPECT_EQ(llvm::toString(Res.takeError()),
            "maximum number of blocks processed");
}

// Regression test for joins of bool-typed lvalue expressions. The first loop
// results in two passes through the code that follows. Each pass results in a
// different `StorageLocation` for the pointee of `v`. Then, the second loop
// causes a join at the loop head where the two environments map expresssion
// `*v` to different `StorageLocation`s.
//
// An earlier version crashed for this condition (for boolean-typed lvalues), so
// this test only verifies that the analysis runs successfully, without
// examining any details of the results.
TEST_F(DataflowAnalysisTest, JoinBoolLValues) {
  std::string Code = R"(
    void target() {
      for (int x = 1; x; x = 0)
        (void)x;
      bool *v;
      if (*v)
        for (int x = 1; x; x = 0)
          (void)x;
    }
  )";
  ASSERT_THAT_ERROR(
      runAnalysis<NoopAnalysis>(Code,
                                [](ASTContext &C) {
                                  auto EnableBuiltIns = DataflowAnalysisOptions{
                                      DataflowAnalysisContext::Options{}};
                                  return NoopAnalysis(C, EnableBuiltIns);
                                })
          .takeError(),
      llvm::Succeeded());
}

struct FunctionCallLattice {
  using FunctionSet = llvm::SmallSet<std::string, 8>;
  FunctionSet CalledFunctions;

  bool operator==(const FunctionCallLattice &Other) const {
    return CalledFunctions == Other.CalledFunctions;
  }

  LatticeJoinEffect join(const FunctionCallLattice &Other) {
    if (Other.CalledFunctions.empty())
      return LatticeJoinEffect::Unchanged;
    const size_t size_before = CalledFunctions.size();
    CalledFunctions.insert(Other.CalledFunctions.begin(),
                           Other.CalledFunctions.end());
    return CalledFunctions.size() == size_before ? LatticeJoinEffect::Unchanged
                                                 : LatticeJoinEffect::Changed;
  }
};

std::ostream &operator<<(std::ostream &OS, const FunctionCallLattice &L) {
  std::string S;
  llvm::raw_string_ostream ROS(S);
  llvm::interleaveComma(L.CalledFunctions, ROS);
  return OS << "{" << S << "}";
}

class FunctionCallAnalysis
    : public DataflowAnalysis<FunctionCallAnalysis, FunctionCallLattice> {
public:
  explicit FunctionCallAnalysis(ASTContext &Context)
      : DataflowAnalysis<FunctionCallAnalysis, FunctionCallLattice>(Context) {}

  static FunctionCallLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, FunctionCallLattice &E, Environment &) {
    auto CS = Elt.getAs<CFGStmt>();
    if (!CS)
      return;
    const auto *S = CS->getStmt();
    if (auto *C = dyn_cast<CallExpr>(S)) {
      if (auto *F = dyn_cast<FunctionDecl>(C->getCalleeDecl())) {
        E.CalledFunctions.insert(F->getNameInfo().getAsString());
      }
    }
  }
};

class NoreturnDestructorTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Expectations) {
    tooling::FileContentMappings FilesContents;
    FilesContents.push_back(std::make_pair<std::string, std::string>(
        "noreturn_destructor_test_defs.h", R"(
      int foo();

      class Fatal {
       public:
        ~Fatal() __attribute__((noreturn));
        int bar();
        int baz();
      };

      class NonFatal {
       public:
        ~NonFatal();
        int bar();
      };
    )"));

    ASSERT_THAT_ERROR(
        test::checkDataflow<FunctionCallAnalysis>(
            AnalysisInputs<FunctionCallAnalysis>(
                Code, ast_matchers::hasName("target"),
                [](ASTContext &C, Environment &) {
                  return FunctionCallAnalysis(C);
                })
                .withASTBuildArgs({"-fsyntax-only", "-std=c++17"})
                .withASTBuildVirtualMappedFiles(std::move(FilesContents)),
            /*VerifyResults=*/
            [&Expectations](
                const llvm::StringMap<
                    DataflowAnalysisState<FunctionCallLattice>> &Results,
                const AnalysisOutputs &) {
              EXPECT_THAT(Results, Expectations);
            }),
        llvm::Succeeded());
  }
};

MATCHER_P(HoldsFunctionCallLattice, m,
          ((negation ? "doesn't hold" : "holds") +
           llvm::StringRef(" a lattice element that ") +
           DescribeMatcher<FunctionCallLattice>(m))
              .str()) {
  return ExplainMatchResult(m, arg.Lattice, result_listener);
}

MATCHER_P(HasCalledFunctions, m,
          ((negation ? "doesn't hold" : "holds") +
           llvm::StringRef(" a set of called functions that ") +
           DescribeMatcher<FunctionCallLattice::FunctionSet>(m))
              .str()) {
  return ExplainMatchResult(m, arg.CalledFunctions, result_listener);
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorBothBranchesReturn) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b) {
      int value = b ? foo() : NonFatal().bar();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsFunctionCallLattice(HasCalledFunctions(
                                 UnorderedElementsAre("foo", "bar"))))));
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorLeftBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b) {
      int value = b ? foo() : Fatal().bar();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsFunctionCallLattice(HasCalledFunctions(
                                 UnorderedElementsAre("foo"))))));
}

TEST_F(NoreturnDestructorTest,
       ConditionalOperatorConstantCondition_LeftBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target() {
      int value = true ? foo() : Fatal().bar();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsFunctionCallLattice(HasCalledFunctions(
                                 UnorderedElementsAre("foo"))))));
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorRightBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b) {
      int value = b ? Fatal().bar() : foo();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsFunctionCallLattice(HasCalledFunctions(
                                 UnorderedElementsAre("foo"))))));
}

TEST_F(NoreturnDestructorTest,
       ConditionalOperatorConstantCondition_RightBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target() {
      int value = false ? Fatal().bar() : foo();
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsFunctionCallLattice(HasCalledFunctions(
                                 UnorderedElementsAre("foo"))))));
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorNestedBranchesDoNotReturn) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b1, bool b2) {
      int value = b1 ? foo() : (b2 ? Fatal().bar() : Fatal().baz());
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, IsEmpty());
  // FIXME: Called functions at point `p` should contain "foo".
}

TEST_F(NoreturnDestructorTest, ConditionalOperatorNestedBranchReturns) {
  std::string Code = R"(
    #include "noreturn_destructor_test_defs.h"

    void target(bool b1, bool b2) {
      int value = b1 ? Fatal().bar() : (b2 ? Fatal().baz() : foo());
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code, UnorderedElementsAre(IsStringMapEntry(
                        "p", HoldsFunctionCallLattice(HasCalledFunctions(
                                 UnorderedElementsAre("baz", "foo"))))));
  // FIXME: Called functions at point `p` should contain only "foo".
}

// Models an analysis that uses flow conditions.
class SpecialBoolAnalysis final
    : public DataflowAnalysis<SpecialBoolAnalysis, NoopLattice> {
public:
  explicit SpecialBoolAnalysis(ASTContext &Context, Environment &Env)
      : DataflowAnalysis<SpecialBoolAnalysis, NoopLattice>(Context) {
    Env.getDataflowAnalysisContext().setSyntheticFieldCallback(
        [](QualType Ty) -> llvm::StringMap<QualType> {
          RecordDecl *RD = Ty->getAsRecordDecl();
          if (RD == nullptr || RD->getIdentifier() == nullptr ||
              RD->getName() != "SpecialBool")
            return {};
          return {{"is_set", RD->getASTContext().BoolTy}};
        });
  }

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, NoopLattice &, Environment &Env) {
    auto CS = Elt.getAs<CFGStmt>();
    if (!CS)
      return;
    const auto *S = CS->getStmt();
    auto SpecialBoolRecordDecl = recordDecl(hasName("SpecialBool"));
    auto HasSpecialBoolType = hasType(SpecialBoolRecordDecl);

    if (const auto *E = selectFirst<CXXConstructExpr>(
            "call", match(cxxConstructExpr(HasSpecialBoolType).bind("call"), *S,
                          getASTContext()))) {
      Env.setValue(Env.getResultObjectLocation(*E).getSyntheticField("is_set"),
                   Env.getBoolLiteralValue(false));
    } else if (const auto *E = selectFirst<CXXMemberCallExpr>(
                   "call", match(cxxMemberCallExpr(callee(cxxMethodDecl(ofClass(
                                                       SpecialBoolRecordDecl))))
                                     .bind("call"),
                                 *S, getASTContext()))) {
      if (RecordStorageLocation *ObjectLoc = getImplicitObjectLocation(*E, Env))
        Env.setValue(ObjectLoc->getSyntheticField("is_set"),
                     Env.getBoolLiteralValue(true));
    }
  }
};

class JoinFlowConditionsTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    ASSERT_THAT_ERROR(
        test::checkDataflow<SpecialBoolAnalysis>(
            AnalysisInputs<SpecialBoolAnalysis>(
                Code, ast_matchers::hasName("target"),
                [](ASTContext &Context, Environment &Env) {
                  return SpecialBoolAnalysis(Context, Env);
                })
                .withASTBuildArgs({"-fsyntax-only", "-std=c++17"}),
            /*VerifyResults=*/[&Match](const llvm::StringMap<
                                           DataflowAnalysisState<NoopLattice>>
                                           &Results,
                                       const AnalysisOutputs
                                           &AO) { Match(Results, AO.ASTCtx); }),
        llvm::Succeeded());
  }
};

TEST_F(JoinFlowConditionsTest, JoinDistinctButProvablyEquivalentValues) {
  std::string Code = R"(
    struct SpecialBool {
      SpecialBool() = default;
      void set();
    };

    void target(bool Cond) {
      SpecialBool Foo;
      /*[[p1]]*/
      if (Cond) {
        Foo.set();
        /*[[p2]]*/
      } else {
        Foo.set();
        /*[[p3]]*/
      }
      (void)0;
      /*[[p4]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("p1", "p2", "p3", "p4"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        const Environment &Env3 = getEnvironmentAtAnnotation(Results, "p3");
        const Environment &Env4 = getEnvironmentAtAnnotation(Results, "p4");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFoo = [FooDecl](const Environment &Env) -> const Formula & {
          auto *Loc =
              cast<RecordStorageLocation>(Env.getStorageLocation(*FooDecl));
          return cast<BoolValue>(Env.getValue(Loc->getSyntheticField("is_set")))
              ->formula();
        };

        EXPECT_FALSE(Env1.proves(GetFoo(Env1)));
        EXPECT_TRUE(Env2.proves(GetFoo(Env2)));
        EXPECT_TRUE(Env3.proves(GetFoo(Env3)));
        EXPECT_TRUE(Env4.proves(GetFoo(Env4)));
      });
}

class NullPointerAnalysis final
    : public DataflowAnalysis<NullPointerAnalysis, NoopLattice> {
public:
  explicit NullPointerAnalysis(ASTContext &Context)
      : DataflowAnalysis<NullPointerAnalysis, NoopLattice>(Context) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, NoopLattice &, Environment &Env) {
    auto CS = Elt.getAs<CFGStmt>();
    if (!CS)
      return;
    const Stmt *S = CS->getStmt();
    const Expr *E = dyn_cast<Expr>(S);
    if (!E)
      return;

    if (!E->getType()->isPointerType())
      return;

    // Make sure we have a `PointerValue` for `E`.
    auto *PtrVal = cast_or_null<PointerValue>(Env.getValue(*E));
    if (PtrVal == nullptr) {
      PtrVal = cast<PointerValue>(Env.createValue(E->getType()));
      Env.setValue(*E, *PtrVal);
    }

    if (auto *Cast = dyn_cast<ImplicitCastExpr>(E);
        Cast && Cast->getCastKind() == CK_NullToPointer)
      PtrVal->setProperty("is_null", Env.getBoolLiteralValue(true));
    else if (auto *Op = dyn_cast<UnaryOperator>(E);
             Op && Op->getOpcode() == UO_AddrOf)
      PtrVal->setProperty("is_null", Env.getBoolLiteralValue(false));
  }

  ComparisonResult compare(QualType Type, const Value &Val1,
                           const Environment &Env1, const Value &Val2,
                           const Environment &Env2) override {
    // Nothing to say about a value that is not a pointer.
    if (!Type->isPointerType())
      return ComparisonResult::Unknown;

    auto *Prop1 = Val1.getProperty("is_null");
    auto *Prop2 = Val2.getProperty("is_null");
    assert(Prop1 != nullptr && Prop2 != nullptr);
    return areEquivalentValues(*Prop1, *Prop2) ? ComparisonResult::Same
                                               : ComparisonResult::Different;
  }

  void join(QualType Type, const Value &Val1, const Environment &Env1,
            const Value &Val2, const Environment &Env2, Value &JoinedVal,
            Environment &JoinedEnv) override {
    // Nothing to say about a value that is not a pointer...
    if (!Type->isPointerType())
      return;

    // ... or, a pointer without the `is_null` property.
    auto *IsNull1 = cast_or_null<BoolValue>(Val1.getProperty("is_null"));
    auto *IsNull2 = cast_or_null<BoolValue>(Val2.getProperty("is_null"));
    if (IsNull1 == nullptr || IsNull2 == nullptr)
      return;

    if (IsNull1 == IsNull2)
      JoinedVal.setProperty("is_null", *IsNull1);
    else
      JoinedVal.setProperty("is_null", JoinedEnv.makeTopBoolValue());
  }
};

class WideningTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    ASSERT_THAT_ERROR(
        checkDataflow<NullPointerAnalysis>(
            AnalysisInputs<NullPointerAnalysis>(
                Code, ast_matchers::hasName("target"),
                [](ASTContext &Context, Environment &Env) {
                  return NullPointerAnalysis(Context);
                })
                .withASTBuildArgs({"-fsyntax-only", "-std=c++17"}),
            /*VerifyResults=*/[&Match](const llvm::StringMap<
                                           DataflowAnalysisState<NoopLattice>>
                                           &Results,
                                       const AnalysisOutputs
                                           &AO) { Match(Results, AO.ASTCtx); }),
        llvm::Succeeded());
  }
};

TEST_F(WideningTest, JoinDistinctValuesWithDistinctProperties) {
  std::string Code = R"(
    void target(bool Cond) {
      int *Foo = nullptr;
      int i = 0;
      /*[[p1]]*/
      if (Cond) {
        Foo = &i;
        /*[[p2]]*/
      }
      (void)0;
      /*[[p3]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2", "p3"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        const Environment &Env3 = getEnvironmentAtAnnotation(Results, "p3");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        EXPECT_EQ(GetFooValue(Env1)->getProperty("is_null"),
                  &Env1.getBoolLiteralValue(true));
        EXPECT_EQ(GetFooValue(Env2)->getProperty("is_null"),
                  &Env2.getBoolLiteralValue(false));
        EXPECT_TRUE(
            isa<TopBoolValue>(GetFooValue(Env3)->getProperty("is_null")));
      });
}

TEST_F(WideningTest, JoinDistinctValuesWithSameProperties) {
  std::string Code = R"(
    void target(bool Cond) {
      int *Foo = nullptr;
      int i1 = 0;
      int i2 = 0;
      /*[[p1]]*/
      if (Cond) {
        Foo = &i1;
        /*[[p2]]*/
      } else {
        Foo = &i2;
        /*[[p3]]*/
      }
      (void)0;
      /*[[p4]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("p1", "p2", "p3", "p4"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        const Environment &Env3 = getEnvironmentAtAnnotation(Results, "p3");
        const Environment &Env4 = getEnvironmentAtAnnotation(Results, "p4");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        EXPECT_EQ(GetFooValue(Env1)->getProperty("is_null"),
                  &Env1.getBoolLiteralValue(true));
        EXPECT_EQ(GetFooValue(Env2)->getProperty("is_null"),
                  &Env2.getBoolLiteralValue(false));
        EXPECT_EQ(GetFooValue(Env3)->getProperty("is_null"),
                  &Env3.getBoolLiteralValue(false));
        EXPECT_EQ(GetFooValue(Env4)->getProperty("is_null"),
                  &Env4.getBoolLiteralValue(false));
      });
}

TEST_F(WideningTest, DistinctPointersToTheSameLocationAreEquivalent) {
  std::string Code = R"(
    void target(int Foo, bool Cond) {
      int *Bar = &Foo;
      while (Cond) {
        Bar = &Foo;
      }
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc =
            cast<ScalarStorageLocation>(Env.getStorageLocation(*FooDecl));
        const auto *BarVal = cast<PointerValue>(Env.getValue(*BarDecl));
        EXPECT_EQ(&BarVal->getPointeeLoc(), FooLoc);
      });
}

TEST_F(WideningTest, DistinctValuesWithSamePropertiesAreEquivalent) {
  std::string Code = R"(
    void target(bool Cond) {
      int *Foo;
      int i1 = 0;
      int i2 = 0;
      Foo = &i1;
      while (Cond) {
        Foo = &i2;
      }
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const auto *FooVal = Env.getValue(*FooDecl);
        EXPECT_EQ(FooVal->getProperty("is_null"),
                  &Env.getBoolLiteralValue(false));
      });
}

class FlowConditionTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    ASSERT_THAT_ERROR(
        checkDataflow<NoopAnalysis>(
            AnalysisInputs<NoopAnalysis>(
                Code, ast_matchers::hasName("target"),
                [](ASTContext &Context, Environment &Env) {
                  return NoopAnalysis(Context);
                })
                .withASTBuildArgs({"-fsyntax-only", "-std=c++17"}),
            /*VerifyResults=*/[&Match](const llvm::StringMap<
                                           DataflowAnalysisState<NoopLattice>>
                                           &Results,
                                       const AnalysisOutputs
                                           &AO) { Match(Results, AO.ASTCtx); }),
        llvm::Succeeded());
  }
};

TEST_F(FlowConditionTest, IfStmtSingleVar) {
  std::string Code = R"(
    void target(bool Foo) {
      if (Foo) {
        (void)0;
        /*[[p1]]*/
      } else {
        (void)1;
        /*[[p2]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));

        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        auto &FooVal1 = cast<BoolValue>(Env1.getValue(*FooDecl))->formula();
        EXPECT_TRUE(Env1.proves(FooVal1));

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
        EXPECT_FALSE(Env2.proves(FooVal2));
      });
}

TEST_F(FlowConditionTest, IfStmtSingleNegatedVar) {
  std::string Code = R"(
    void target(bool Foo) {
      if (!Foo) {
        (void)0;
        /*[[p1]]*/
      } else {
        (void)1;
        /*[[p2]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));

        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        auto &FooVal1 = cast<BoolValue>(Env1.getValue(*FooDecl))->formula();
        EXPECT_FALSE(Env1.proves(FooVal1));

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
        EXPECT_TRUE(Env2.proves(FooVal2));
      });
}

TEST_F(FlowConditionTest, WhileStmt) {
  std::string Code = R"(
    void target(bool Foo) {
      while (Foo) {
        (void)0;
        /*[[p]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        auto &FooVal = cast<BoolValue>(Env.getValue(*FooDecl))->formula();
        EXPECT_TRUE(Env.proves(FooVal));
      });
}

TEST_F(FlowConditionTest, WhileStmtWithAssignmentInCondition) {
  std::string Code = R"(
    void target(bool Foo) {
      // This test checks whether the analysis preserves the connection between
      // the value of `Foo` and the assignment expression, despite widening.
      // The equality operator generates a fresh boolean variable on each
      // interpretation, which forces use of widening.
      while ((Foo = (3 == 4))) {
        (void)0;
        /*[[p]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &FooVal = getValueForDecl<BoolValue>(ASTCtx, Env, "Foo").formula();
        EXPECT_TRUE(Env.proves(FooVal));
      });
}

TEST_F(FlowConditionTest, Conjunction) {
  std::string Code = R"(
    void target(bool Foo, bool Bar) {
      if (Foo && Bar) {
        (void)0;
        /*[[p1]]*/
      } else {
        (void)1;
        /*[[p2]]*/
      }
    }
  )";
  runDataflow(Code, [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                           &Results,
                       ASTContext &ASTCtx) {
    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
    ASSERT_THAT(BarDecl, NotNull());

    ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));

    const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
    auto &FooVal1 = cast<BoolValue>(Env1.getValue(*FooDecl))->formula();
    auto &BarVal1 = cast<BoolValue>(Env1.getValue(*BarDecl))->formula();
    EXPECT_TRUE(Env1.proves(FooVal1));
    EXPECT_TRUE(Env1.proves(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env2.proves(FooVal2));
    EXPECT_FALSE(Env2.proves(BarVal2));
  });
}

TEST_F(FlowConditionTest, Disjunction) {
  std::string Code = R"(
    void target(bool Foo, bool Bar) {
      if (Foo || Bar) {
        (void)0;
        /*[[p1]]*/
      } else {
        (void)1;
        /*[[p2]]*/
      }
    }
  )";
  runDataflow(Code, [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                           &Results,
                       ASTContext &ASTCtx) {
    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
    ASSERT_THAT(BarDecl, NotNull());

    ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));

    const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
    auto &FooVal1 = cast<BoolValue>(Env1.getValue(*FooDecl))->formula();
    auto &BarVal1 = cast<BoolValue>(Env1.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env1.proves(FooVal1));
    EXPECT_FALSE(Env1.proves(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env2.proves(FooVal2));
    EXPECT_FALSE(Env2.proves(BarVal2));
  });
}

TEST_F(FlowConditionTest, NegatedConjunction) {
  std::string Code = R"(
    void target(bool Foo, bool Bar) {
      if (!(Foo && Bar)) {
        (void)0;
        /*[[p1]]*/
      } else {
        (void)1;
        /*[[p2]]*/
      }
    }
  )";
  runDataflow(Code, [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                           &Results,
                       ASTContext &ASTCtx) {
    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
    ASSERT_THAT(BarDecl, NotNull());

    ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));

    const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
    auto &FooVal1 = cast<BoolValue>(Env1.getValue(*FooDecl))->formula();
    auto &BarVal1 = cast<BoolValue>(Env1.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env1.proves(FooVal1));
    EXPECT_FALSE(Env1.proves(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_TRUE(Env2.proves(FooVal2));
    EXPECT_TRUE(Env2.proves(BarVal2));
  });
}

TEST_F(FlowConditionTest, DeMorgan) {
  std::string Code = R"(
    void target(bool Foo, bool Bar) {
      if (!(!Foo || !Bar)) {
        (void)0;
        /*[[p1]]*/
      } else {
        (void)1;
        /*[[p2]]*/
      }
    }
  )";
  runDataflow(Code, [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>>
                           &Results,
                       ASTContext &ASTCtx) {
    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
    ASSERT_THAT(BarDecl, NotNull());

    ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));

    const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
    auto &FooVal1 = cast<BoolValue>(Env1.getValue(*FooDecl))->formula();
    auto &BarVal1 = cast<BoolValue>(Env1.getValue(*BarDecl))->formula();
    EXPECT_TRUE(Env1.proves(FooVal1));
    EXPECT_TRUE(Env1.proves(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env2.proves(FooVal2));
    EXPECT_FALSE(Env2.proves(BarVal2));
  });
}

TEST_F(FlowConditionTest, Join) {
  std::string Code = R"(
    void target(bool Foo, bool Bar) {
      if (Bar) {
        if (!Foo)
          return;
      } else {
        if (!Foo)
          return;
      }
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");
        auto &FooVal = cast<BoolValue>(Env.getValue(*FooDecl))->formula();
        EXPECT_TRUE(Env.proves(FooVal));
      });
}

// Verifies that flow conditions are properly constructed even when the
// condition is not meaningfully interpreted.
//
// Note: currently, arbitrary function calls are uninterpreted, so the test
// exercises this case. If and when we change that, this test will not add to
// coverage (although it may still test a valuable case).
TEST_F(FlowConditionTest, OpaqueFlowConditionJoinsToOpaqueBool) {
  std::string Code = R"(
    bool foo();

    void target() {
      bool Bar = true;
      if (foo())
        Bar = false;
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal = cast<BoolValue>(Env.getValue(*BarDecl))->formula();

        EXPECT_FALSE(Env.proves(BarVal));
      });
}

// Verifies that flow conditions are properly constructed even when the
// condition is not meaningfully interpreted.
//
// Note: currently, fields with recursive type calls are uninterpreted (beneath
// the first instance), so the test exercises this case. If and when we change
// that, this test will not add to coverage (although it may still test a
// valuable case).
TEST_F(FlowConditionTest, OpaqueFieldFlowConditionJoinsToOpaqueBool) {
  std::string Code = R"(
    struct Rec {
      Rec* Next;
    };

    struct Foo {
      Rec* X;
    };

    void target(Foo F) {
      bool Bar = true;
      if (F.X->Next)
        Bar = false;
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal = cast<BoolValue>(Env.getValue(*BarDecl))->formula();

        EXPECT_FALSE(Env.proves(BarVal));
      });
}

// Verifies that flow conditions are properly constructed even when the
// condition is not meaningfully interpreted. Adds to above by nesting the
// interestnig case inside a normal branch. This protects against degenerate
// solutions which only test for empty flow conditions, for example.
TEST_F(FlowConditionTest, OpaqueFlowConditionInsideBranchJoinsToOpaqueBool) {
  std::string Code = R"(
    bool foo();

    void target(bool Cond) {
      bool Bar = true;
      if (Cond) {
        if (foo())
          Bar = false;
        (void)0;
        /*[[p]]*/
      }
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal = cast<BoolValue>(Env.getValue(*BarDecl))->formula();

        EXPECT_FALSE(Env.proves(BarVal));
      });
}

TEST_F(FlowConditionTest, PointerToBoolImplicitCast) {
  std::string Code = R"(
    void target(int *Ptr) {
      bool Foo = false;
      if (Ptr) {
        Foo = true;
        /*[[p1]]*/
      }

      (void)0;
      /*[[p2]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         ASTContext &ASTCtx) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        auto &FooVal1 = cast<BoolValue>(Env1.getValue(*FooDecl))->formula();
        EXPECT_TRUE(Env1.proves(FooVal1));

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
        EXPECT_FALSE(Env2.proves(FooVal2));
      });
}

class TopAnalysis final : public DataflowAnalysis<TopAnalysis, NoopLattice> {
public:
  explicit TopAnalysis(ASTContext &Context)
      : DataflowAnalysis<TopAnalysis, NoopLattice>(Context) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, NoopLattice &, Environment &Env) {
    auto CS = Elt.getAs<CFGStmt>();
    if (!CS)
      return;
    const Stmt *S = CS->getStmt();
    SmallVector<BoundNodes, 1> Matches =
        match(callExpr(callee(functionDecl(hasName("makeTop")))).bind("top"),
              *S, getASTContext());
    if (const auto *E = selectFirst<CallExpr>("top", Matches)) {
      Env.setValue(*E, Env.makeTopBoolValue());
    }
  }

  ComparisonResult compare(QualType Type, const Value &Val1,
                           const Environment &Env1, const Value &Val2,
                           const Environment &Env2) override {
    // Changes to a sound approximation, which allows us to test whether we can
    // (soundly) converge for some loops.
    return ComparisonResult::Unknown;
  }
};

class TopTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher VerifyResults) {
    ASSERT_THAT_ERROR(
        checkDataflow<TopAnalysis>(
            AnalysisInputs<TopAnalysis>(
                Code, ast_matchers::hasName("target"),
                [](ASTContext &Context, Environment &Env) {
                  return TopAnalysis(Context);
                })
                .withASTBuildArgs({"-fsyntax-only", "-std=c++17"}),
            VerifyResults),
        llvm::Succeeded());
  }
};

// Tests that when Top is unused it remains Top.
TEST_F(TopTest, UnusedTopInitializer) {
  std::string Code = R"(
    bool makeTop();

    void target() {
      bool Foo = makeTop();
      /*[[p1]]*/
      (void)0;
      /*[[p2]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         const AnalysisOutputs &AO) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(AO.ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        Value *FooVal1 = GetFooValue(Env1);
        ASSERT_THAT(FooVal1, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal1))
            << debugString(FooVal1->getKind());

        Value *FooVal2 = GetFooValue(Env2);
        ASSERT_THAT(FooVal2, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal2))
            << debugString(FooVal2->getKind());

        EXPECT_EQ(FooVal1, FooVal2);
      });
}

// Tests that when Top is unused it remains Top. Like above, but uses the
// assignment form rather than initialization, which uses Top as an lvalue that
// is *not* in an rvalue position.
TEST_F(TopTest, UnusedTopAssignment) {
  std::string Code = R"(
    bool makeTop();

    void target() {
      bool Foo;
      Foo = makeTop();
      /*[[p1]]*/
      (void)0;
      /*[[p2]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         const AnalysisOutputs &AO) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(AO.ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        Value *FooVal1 = GetFooValue(Env1);
        ASSERT_THAT(FooVal1, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal1))
            << debugString(FooVal1->getKind());

        Value *FooVal2 = GetFooValue(Env2);
        ASSERT_THAT(FooVal2, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal2))
            << debugString(FooVal2->getKind());

        EXPECT_EQ(FooVal1, FooVal2);
      });
}

TEST_F(TopTest, UnusedTopJoinsToTop) {
  std::string Code = R"(
    bool makeTop();

    void target(bool Cond, bool F) {
      bool Foo = makeTop();
      // Force a new CFG block.
      if (F) return;
      (void)0;
      /*[[p1]]*/

      bool Zab1;
      bool Zab2;
      if (Cond) {
        Zab1 = true;
      } else {
        Zab2 = true;
      }
      (void)0;
      /*[[p2]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         const AnalysisOutputs &AO) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(AO.ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        Value *FooVal1 = GetFooValue(Env1);
        ASSERT_THAT(FooVal1, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal1))
            << debugString(FooVal1->getKind());

        Value *FooVal2 = GetFooValue(Env2);
        ASSERT_THAT(FooVal2, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal2))
            << debugString(FooVal2->getKind());
      });
}

TEST_F(TopTest, TopUsedBeforeBranchJoinsToSameAtomicBool) {
  std::string Code = R"(
    bool makeTop();

    void target(bool Cond, bool F) {
      bool Foo = makeTop();
      /*[[p0]]*/

      // Use `Top`.
      bool Bar = Foo;
      // Force a new CFG block.
      if (F) return;
      (void)0;
      /*[[p1]]*/

      bool Zab1;
      bool Zab2;
      if (Cond) {
        Zab1 = true;
      } else {
        Zab2 = true;
      }
      (void)0;
      /*[[p2]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         const AnalysisOutputs &AO) {
        ASSERT_THAT(Results.keys(),
                    UnorderedElementsAre("p0", "p1", "p2"));
        const Environment &Env0 = getEnvironmentAtAnnotation(Results, "p0");
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(AO.ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        Value *FooVal0 = GetFooValue(Env0);
        ASSERT_THAT(FooVal0, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal0))
            << debugString(FooVal0->getKind());

        Value *FooVal1 = GetFooValue(Env1);
        ASSERT_THAT(FooVal1, NotNull());
        EXPECT_TRUE(isa<AtomicBoolValue>(FooVal1))
            << debugString(FooVal1->getKind());

        Value *FooVal2 = GetFooValue(Env2);
        ASSERT_THAT(FooVal2, NotNull());
        EXPECT_TRUE(isa<AtomicBoolValue>(FooVal2))
            << debugString(FooVal2->getKind());

        EXPECT_EQ(FooVal2, FooVal1);
      });
}

TEST_F(TopTest, TopUsedInBothBranchesJoinsToAtomic) {
  std::string Code = R"(
    bool makeTop();

    void target(bool Cond, bool F) {
      bool Foo = makeTop();
      // Force a new CFG block.
      if (F) return;
      (void)0;
      /*[[p1]]*/

      bool Zab1;
      bool Zab2;
      if (Cond) {
        Zab1 = Foo;
      } else {
        Zab2 = Foo;
      }
      (void)0;
      /*[[p2]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         const AnalysisOutputs &AO) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(AO.ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        Value *FooVal1 = GetFooValue(Env1);
        ASSERT_THAT(FooVal1, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal1))
            << debugString(FooVal1->getKind());

        Value *FooVal2 = GetFooValue(Env2);
        ASSERT_THAT(FooVal2, NotNull());
        EXPECT_TRUE(isa<AtomicBoolValue>(FooVal2))
            << debugString(FooVal2->getKind());
      });
}

TEST_F(TopTest, TopUsedInBothBranchesWithoutPrecisionLoss) {
  std::string Code = R"(
    bool makeTop();

    void target(bool Cond, bool F) {
      bool Foo = makeTop();
      // Force a new CFG block.
      if (F) return;
      (void)0;

      bool Bar;
      if (Cond) {
        Bar = Foo;
      } else {
        Bar = Foo;
      }
      (void)0;
      /*[[p]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         const AnalysisOutputs &AO) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
        const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

        const ValueDecl *FooDecl = findValueDecl(AO.ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const ValueDecl *BarDecl = findValueDecl(AO.ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto *FooVal = dyn_cast_or_null<BoolValue>(Env.getValue(*FooDecl));
        ASSERT_THAT(FooVal, NotNull());

        auto *BarVal = dyn_cast_or_null<BoolValue>(Env.getValue(*BarDecl));
        ASSERT_THAT(BarVal, NotNull());

        EXPECT_TRUE(Env.proves(
            Env.arena().makeEquals(FooVal->formula(), BarVal->formula())));
      });
}

TEST_F(TopTest, TopUnusedBeforeLoopHeadJoinsToTop) {
  std::string Code = R"(
    bool makeTop();

    void target(bool Cond, bool F) {
      bool Foo = makeTop();
      // Force a new CFG block.
      if (F) return;
      (void)0;
      /*[[p1]]*/

      while (Cond) {
        // Use `Foo`.
        bool Zab = Foo;
        Zab = false;
        Foo = makeTop();
      }
      (void)0;
      /*[[p2]]*/
    }
  )";
  runDataflow(
      Code,
      [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &Results,
         const AnalysisOutputs &AO) {
        ASSERT_THAT(Results.keys(), UnorderedElementsAre("p1", "p2"));
        const Environment &Env1 = getEnvironmentAtAnnotation(Results, "p1");
        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");

        const ValueDecl *FooDecl = findValueDecl(AO.ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return Env.getValue(*FooDecl);
        };

        Value *FooVal1 = GetFooValue(Env1);
        ASSERT_THAT(FooVal1, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal1))
            << debugString(FooVal1->getKind());

        Value *FooVal2 = GetFooValue(Env2);
        ASSERT_THAT(FooVal2, NotNull());
        EXPECT_TRUE(isa<TopBoolValue>(FooVal2))
            << debugString(FooVal2->getKind());

      });
}

TEST_F(TopTest, ForRangeStmtConverges) {
  std::string Code = R"(
    void target(bool Foo) {
      int Ints[10];
      bool B = false;
      for (int I : Ints)
        B = true;
    }
  )";
  runDataflow(Code,
              [](const llvm::StringMap<DataflowAnalysisState<NoopLattice>> &,
                 const AnalysisOutputs &) {
                // No additional expectations. We're only checking that the
                // analysis converged.
              });
}
} // namespace
