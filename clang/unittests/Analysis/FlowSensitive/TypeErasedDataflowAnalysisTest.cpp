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

template <typename AnalysisT>
llvm::Expected<std::vector<
    std::optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>>
runAnalysis(llvm::StringRef Code, AnalysisT (*MakeAnalysis)(ASTContext &)) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-std=c++11"});

  auto *Func = selectFirst<FunctionDecl>(
      "func", match(functionDecl(ast_matchers::hasName("target")).bind("func"),
                    AST->getASTContext()));
  assert(Func != nullptr);

  auto CFCtx =
      llvm::cantFail(ControlFlowContext::build(*Func));

  AnalysisT Analysis = MakeAnalysis(AST->getASTContext());
  DataflowAnalysisContext DACtx(std::make_unique<WatchedLiteralsSolver>());
  Environment Env(DACtx);

  return runDataflowAnalysis(CFCtx, Analysis, Env);
}

TEST(DataflowAnalysisTest, NoopAnalysis) {
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
TEST(DataflowAnalysisTest, DiagnoseFunctionDiagnoserCalledOnEachElement) {
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

TEST(DataflowAnalysisTest, NonConvergingAnalysis) {
  std::string Code = R"(
    void target() {
      while(true) {}
    }
  )";
  auto Res = runAnalysis<NonConvergingAnalysis>(
      Code, [](ASTContext &C) { return NonConvergingAnalysis(C); });
  EXPECT_EQ(llvm::toString(Res.takeError()),
            "maximum number of iterations reached");
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
TEST(DataflowAnalysisTest, JoinBoolLValues) {
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
  explicit SpecialBoolAnalysis(ASTContext &Context)
      : DataflowAnalysis<SpecialBoolAnalysis, NoopLattice>(Context) {}

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
      cast<RecordValue>(Env.getValue(*E))
          ->setProperty("is_set", Env.getBoolLiteralValue(false));
    } else if (const auto *E = selectFirst<CXXMemberCallExpr>(
                   "call", match(cxxMemberCallExpr(callee(cxxMethodDecl(ofClass(
                                                       SpecialBoolRecordDecl))))
                                     .bind("call"),
                                 *S, getASTContext()))) {
      auto &ObjectLoc =
          *cast<RecordStorageLocation>(getImplicitObjectLocation(*E, Env));

      refreshRecordValue(ObjectLoc, Env)
          .setProperty("is_set", Env.getBoolLiteralValue(true));
    }
  }

  ComparisonResult compare(QualType Type, const Value &Val1,
                           const Environment &Env1, const Value &Val2,
                           const Environment &Env2) override {
    const auto *Decl = Type->getAsCXXRecordDecl();
    if (Decl == nullptr || Decl->getIdentifier() == nullptr ||
        Decl->getName() != "SpecialBool")
      return ComparisonResult::Unknown;

    auto *IsSet1 = cast_or_null<BoolValue>(Val1.getProperty("is_set"));
    auto *IsSet2 = cast_or_null<BoolValue>(Val2.getProperty("is_set"));
    if (IsSet1 == nullptr)
      return IsSet2 == nullptr ? ComparisonResult::Same
                               : ComparisonResult::Different;

    if (IsSet2 == nullptr)
      return ComparisonResult::Different;

    return Env1.flowConditionImplies(IsSet1->formula()) ==
                   Env2.flowConditionImplies(IsSet2->formula())
               ? ComparisonResult::Same
               : ComparisonResult::Different;
  }

  // Always returns `true` to accept the `MergedVal`.
  bool merge(QualType Type, const Value &Val1, const Environment &Env1,
             const Value &Val2, const Environment &Env2, Value &MergedVal,
             Environment &MergedEnv) override {
    const auto *Decl = Type->getAsCXXRecordDecl();
    if (Decl == nullptr || Decl->getIdentifier() == nullptr ||
        Decl->getName() != "SpecialBool")
      return true;

    auto *IsSet1 = cast_or_null<BoolValue>(Val1.getProperty("is_set"));
    if (IsSet1 == nullptr)
      return true;

    auto *IsSet2 = cast_or_null<BoolValue>(Val2.getProperty("is_set"));
    if (IsSet2 == nullptr)
      return true;

    auto &IsSet = MergedEnv.makeAtomicBoolValue();
    MergedVal.setProperty("is_set", IsSet);
    if (Env1.flowConditionImplies(IsSet1->formula()) &&
        Env2.flowConditionImplies(IsSet2->formula()))
      MergedEnv.addToFlowCondition(IsSet.formula());

    return true;
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
                  return SpecialBoolAnalysis(Context);
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
          return cast<BoolValue>(Env.getValue(*FooDecl)->getProperty("is_set"))
              ->formula();
        };

        EXPECT_FALSE(Env1.flowConditionImplies(GetFoo(Env1)));
        EXPECT_TRUE(Env2.flowConditionImplies(GetFoo(Env2)));
        EXPECT_TRUE(Env3.flowConditionImplies(GetFoo(Env3)));
        EXPECT_TRUE(Env4.flowConditionImplies(GetFoo(Env4)));
      });
}

class OptionalIntAnalysis final
    : public DataflowAnalysis<OptionalIntAnalysis, NoopLattice> {
public:
  explicit OptionalIntAnalysis(ASTContext &Context)
      : DataflowAnalysis<OptionalIntAnalysis, NoopLattice>(Context) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, NoopLattice &, Environment &Env) {
    auto CS = Elt.getAs<CFGStmt>();
    if (!CS)
      return;
    const Stmt *S = CS->getStmt();
    auto OptionalIntRecordDecl = recordDecl(hasName("OptionalInt"));
    auto HasOptionalIntType = hasType(OptionalIntRecordDecl);

    SmallVector<BoundNodes, 1> Matches = match(
        stmt(anyOf(cxxConstructExpr(HasOptionalIntType).bind("construct"),
                   cxxOperatorCallExpr(
                       callee(cxxMethodDecl(ofClass(OptionalIntRecordDecl))))
                       .bind("operator"))),
        *S, getASTContext());
    if (const auto *E = selectFirst<CXXConstructExpr>(
            "construct", Matches)) {
      cast<RecordValue>(Env.getValue(*E))
          ->setProperty("has_value", Env.getBoolLiteralValue(false));
    } else if (const auto *E =
                   selectFirst<CXXOperatorCallExpr>("operator", Matches)) {
      assert(E->getNumArgs() > 0);
      auto *Object = E->getArg(0);
      assert(Object != nullptr);

      refreshRecordValue(*Object, Env)
          .setProperty("has_value", Env.getBoolLiteralValue(true));
    }
  }

  ComparisonResult compare(QualType Type, const Value &Val1,
                           const Environment &Env1, const Value &Val2,
                           const Environment &Env2) override {
    // Nothing to say about a value that does not model an `OptionalInt`.
    if (!Type->isRecordType() ||
        Type->getAsCXXRecordDecl()->getQualifiedNameAsString() != "OptionalInt")
      return ComparisonResult::Unknown;

    auto *Prop1 = Val1.getProperty("has_value");
    auto *Prop2 = Val2.getProperty("has_value");
    assert(Prop1 != nullptr && Prop2 != nullptr);
    return areEquivalentValues(*Prop1, *Prop2) ? ComparisonResult::Same
                                               : ComparisonResult::Different;
  }

  bool merge(QualType Type, const Value &Val1, const Environment &Env1,
             const Value &Val2, const Environment &Env2, Value &MergedVal,
             Environment &MergedEnv) override {
    // Nothing to say about a value that does not model an `OptionalInt`.
    if (!Type->isRecordType() ||
        Type->getAsCXXRecordDecl()->getQualifiedNameAsString() != "OptionalInt")
      return false;

    auto *HasValue1 = cast_or_null<BoolValue>(Val1.getProperty("has_value"));
    if (HasValue1 == nullptr)
      return false;

    auto *HasValue2 = cast_or_null<BoolValue>(Val2.getProperty("has_value"));
    if (HasValue2 == nullptr)
      return false;

    if (HasValue1 == HasValue2)
      MergedVal.setProperty("has_value", *HasValue1);
    else
      MergedVal.setProperty("has_value", MergedEnv.makeTopBoolValue());
    return true;
  }
};

class WideningTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    tooling::FileContentMappings FilesContents;
    FilesContents.push_back(
        std::make_pair<std::string, std::string>("widening_test_defs.h", R"(
      struct OptionalInt {
        OptionalInt() = default;
        OptionalInt& operator=(int);
      };
    )"));
    ASSERT_THAT_ERROR(
        checkDataflow<OptionalIntAnalysis>(
            AnalysisInputs<OptionalIntAnalysis>(
                Code, ast_matchers::hasName("target"),
                [](ASTContext &Context, Environment &Env) {
                  return OptionalIntAnalysis(Context);
                })
                .withASTBuildArgs({"-fsyntax-only", "-std=c++17"})
                .withASTBuildVirtualMappedFiles(std::move(FilesContents)),
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
    #include "widening_test_defs.h"

    void target(bool Cond) {
      OptionalInt Foo;
      /*[[p1]]*/
      if (Cond) {
        Foo = 1;
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

        EXPECT_EQ(GetFooValue(Env1)->getProperty("has_value"),
                  &Env1.getBoolLiteralValue(false));
        EXPECT_EQ(GetFooValue(Env2)->getProperty("has_value"),
                  &Env2.getBoolLiteralValue(true));
        EXPECT_TRUE(
            isa<TopBoolValue>(GetFooValue(Env3)->getProperty("has_value")));
      });
}

TEST_F(WideningTest, JoinDistinctValuesWithSameProperties) {
  std::string Code = R"(
    #include "widening_test_defs.h"

    void target(bool Cond) {
      OptionalInt Foo;
      /*[[p1]]*/
      if (Cond) {
        Foo = 1;
        /*[[p2]]*/
      } else {
        Foo = 2;
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

        EXPECT_EQ(GetFooValue(Env1)->getProperty("has_value"),
                  &Env1.getBoolLiteralValue(false));
        EXPECT_EQ(GetFooValue(Env2)->getProperty("has_value"),
                  &Env2.getBoolLiteralValue(true));
        EXPECT_EQ(GetFooValue(Env3)->getProperty("has_value"),
                  &Env3.getBoolLiteralValue(true));
        EXPECT_EQ(GetFooValue(Env4)->getProperty("has_value"),
                  &Env4.getBoolLiteralValue(true));
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
    #include "widening_test_defs.h"

    void target(bool Cond) {
      OptionalInt Foo;
      Foo = 1;
      while (Cond) {
        Foo = 2;
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
        EXPECT_EQ(FooVal->getProperty("has_value"),
                  &Env.getBoolLiteralValue(true));
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
        EXPECT_TRUE(Env1.flowConditionImplies(FooVal1));

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
        EXPECT_FALSE(Env2.flowConditionImplies(FooVal2));
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
        EXPECT_FALSE(Env1.flowConditionImplies(FooVal1));

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
        EXPECT_TRUE(Env2.flowConditionImplies(FooVal2));
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
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
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
    EXPECT_TRUE(Env1.flowConditionImplies(FooVal1));
    EXPECT_TRUE(Env1.flowConditionImplies(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env2.flowConditionImplies(FooVal2));
    EXPECT_FALSE(Env2.flowConditionImplies(BarVal2));
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
    EXPECT_FALSE(Env1.flowConditionImplies(FooVal1));
    EXPECT_FALSE(Env1.flowConditionImplies(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env2.flowConditionImplies(FooVal2));
    EXPECT_FALSE(Env2.flowConditionImplies(BarVal2));
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
    EXPECT_FALSE(Env1.flowConditionImplies(FooVal1));
    EXPECT_FALSE(Env1.flowConditionImplies(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_TRUE(Env2.flowConditionImplies(FooVal2));
    EXPECT_TRUE(Env2.flowConditionImplies(BarVal2));
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
    EXPECT_TRUE(Env1.flowConditionImplies(FooVal1));
    EXPECT_TRUE(Env1.flowConditionImplies(BarVal1));

    const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
    auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
    auto &BarVal2 = cast<BoolValue>(Env2.getValue(*BarDecl))->formula();
    EXPECT_FALSE(Env2.flowConditionImplies(FooVal2));
    EXPECT_FALSE(Env2.flowConditionImplies(BarVal2));
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
        EXPECT_TRUE(Env.flowConditionImplies(FooVal));
      });
}

// Verifies that flow conditions are properly constructed even when the
// condition is not meaningfully interpreted.
//
// Note: currently, arbitrary function calls are uninterpreted, so the test
// exercises this case. If and when we change that, this test will not add to
// coverage (although it may still test a valuable case).
TEST_F(FlowConditionTest, OpaqueFlowConditionMergesToOpaqueBool) {
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

        EXPECT_FALSE(Env.flowConditionImplies(BarVal));
      });
}

// Verifies that flow conditions are properly constructed even when the
// condition is not meaningfully interpreted.
//
// Note: currently, fields with recursive type calls are uninterpreted (beneath
// the first instance), so the test exercises this case. If and when we change
// that, this test will not add to coverage (although it may still test a
// valuable case).
TEST_F(FlowConditionTest, OpaqueFieldFlowConditionMergesToOpaqueBool) {
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

        EXPECT_FALSE(Env.flowConditionImplies(BarVal));
      });
}

// Verifies that flow conditions are properly constructed even when the
// condition is not meaningfully interpreted. Adds to above by nesting the
// interestnig case inside a normal branch. This protects against degenerate
// solutions which only test for empty flow conditions, for example.
TEST_F(FlowConditionTest, OpaqueFlowConditionInsideBranchMergesToOpaqueBool) {
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

        EXPECT_FALSE(Env.flowConditionImplies(BarVal));
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
        EXPECT_TRUE(Env1.flowConditionImplies(FooVal1));

        const Environment &Env2 = getEnvironmentAtAnnotation(Results, "p2");
        auto &FooVal2 = cast<BoolValue>(Env2.getValue(*FooDecl))->formula();
        EXPECT_FALSE(Env2.flowConditionImplies(FooVal2));
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

        EXPECT_TRUE(Env.flowConditionImplies(
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
