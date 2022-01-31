//===- unittests/Analysis/FlowSensitive/TypeErasedDataflowAnalysisTest.cpp ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoopAnalysis.h"
#include "TestingSupport.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace clang;
using namespace dataflow;
using namespace test;
using namespace ast_matchers;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Pair;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

template <typename AnalysisT>
llvm::Expected<std::vector<
    llvm::Optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>>
runAnalysis(llvm::StringRef Code, AnalysisT (*MakeAnalysis)(ASTContext &)) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-std=c++11"});

  auto *Func = selectFirst<FunctionDecl>(
      "func", match(functionDecl(ast_matchers::hasName("target")).bind("func"),
                    AST->getASTContext()));
  assert(Func != nullptr);

  Stmt *Body = Func->getBody();
  assert(Body != nullptr);

  auto CFCtx = llvm::cantFail(
      ControlFlowContext::build(nullptr, Body, &AST->getASTContext()));

  AnalysisT Analysis = MakeAnalysis(AST->getASTContext());
  DataflowAnalysisContext DACtx;
  Environment Env(DACtx);

  return runDataflowAnalysis(CFCtx, Analysis, Env);
}

TEST(DataflowAnalysisTest, NoopAnalysis) {
  auto BlockStates = llvm::cantFail(
      runAnalysis<NoopAnalysis>("void target() {}", [](ASTContext &C) {
        return NoopAnalysis(C, false);
      }));
  EXPECT_EQ(BlockStates.size(), 2u);
  EXPECT_TRUE(BlockStates[0].hasValue());
  EXPECT_TRUE(BlockStates[1].hasValue());
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
            /*ApplyBuiltinTransfer=*/false) {}

  static NonConvergingLattice initialElement() { return {0}; }

  void transfer(const Stmt *S, NonConvergingLattice &E, Environment &Env) {
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

struct FunctionCallLattice {
  llvm::SmallSet<std::string, 8> CalledFunctions;

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

  void transfer(const Stmt *S, FunctionCallLattice &E, Environment &Env) {
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
            Code, "target",
            [](ASTContext &C, Environment &) {
              return FunctionCallAnalysis(C);
            },
            [&Expectations](
                llvm::ArrayRef<std::pair<
                    std::string, DataflowAnalysisState<FunctionCallLattice>>>
                    Results,
                ASTContext &) { EXPECT_THAT(Results, Expectations); },
            {"-fsyntax-only", "-std=c++17"}, FilesContents),
        llvm::Succeeded());
  }
};

MATCHER_P(HoldsFunctionCallLattice, m,
          ((negation ? "doesn't hold" : "holds") +
           llvm::StringRef(" a lattice element that ") +
           ::testing::DescribeMatcher<FunctionCallLattice>(m, negation))
              .str()) {
  return ExplainMatchResult(m, arg.Lattice, result_listener);
}

MATCHER_P(HasCalledFunctions, m, "") {
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
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
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
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
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
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
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
  runDataflow(Code, UnorderedElementsAre(
                        Pair("p", HoldsFunctionCallLattice(HasCalledFunctions(
                                      UnorderedElementsAre("baz", "foo"))))));
  // FIXME: Called functions at point `p` should contain only "foo".
}

class OptionalIntAnalysis
    : public DataflowAnalysis<OptionalIntAnalysis, NoopLattice> {
public:
  explicit OptionalIntAnalysis(ASTContext &Context, BoolValue &HasValueTop)
      : DataflowAnalysis<OptionalIntAnalysis, NoopLattice>(Context),
        HasValueTop(HasValueTop) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const Stmt *S, NoopLattice &, Environment &Env) {
    auto OptionalIntRecordDecl = recordDecl(hasName("OptionalInt"));
    auto HasOptionalIntType = hasType(OptionalIntRecordDecl);

    if (const auto *E = selectFirst<CXXConstructExpr>(
            "call", match(cxxConstructExpr(HasOptionalIntType).bind("call"), *S,
                          getASTContext()))) {
      auto &ConstructorVal = *cast<StructValue>(Env.createValue(E->getType()));
      ConstructorVal.setProperty("has_value", Env.getBoolLiteralValue(false));
      Env.setValue(*Env.getStorageLocation(*E, SkipPast::None), ConstructorVal);
    } else if (const auto *E = selectFirst<CXXOperatorCallExpr>(
                   "call",
                   match(cxxOperatorCallExpr(callee(cxxMethodDecl(ofClass(
                                                 OptionalIntRecordDecl))))
                             .bind("call"),
                         *S, getASTContext()))) {
      assert(E->getNumArgs() > 0);
      auto *Object = E->getArg(0);
      assert(Object != nullptr);

      auto *ObjectLoc =
          Env.getStorageLocation(*Object, SkipPast::ReferenceThenPointer);
      assert(ObjectLoc != nullptr);

      auto &ConstructorVal =
          *cast<StructValue>(Env.createValue(Object->getType()));
      ConstructorVal.setProperty("has_value", Env.getBoolLiteralValue(true));
      Env.setValue(*ObjectLoc, ConstructorVal);
    }
  }

  bool compareEquivalent(QualType Type, const Value &Val1,
                         const Value &Val2) final {
    // Nothing to say about a value that does not model an `OptionalInt`.
    if (!Type->isRecordType() ||
        Type->getAsCXXRecordDecl()->getQualifiedNameAsString() != "OptionalInt")
      return false;

    return cast<StructValue>(&Val1)->getProperty("has_value") ==
           cast<StructValue>(&Val2)->getProperty("has_value");
  }

  bool merge(QualType Type, const Value &Val1, const Value &Val2,
             Value &MergedVal, Environment &Env) final {
    // Nothing to say about a value that does not model an `OptionalInt`.
    if (!Type->isRecordType() ||
        Type->getAsCXXRecordDecl()->getQualifiedNameAsString() != "OptionalInt")
      return false;

    auto *HasValue1 = cast_or_null<BoolValue>(
        cast<StructValue>(&Val1)->getProperty("has_value"));
    if (HasValue1 == nullptr)
      return false;

    auto *HasValue2 = cast_or_null<BoolValue>(
        cast<StructValue>(&Val2)->getProperty("has_value"));
    if (HasValue2 == nullptr)
      return false;

    assert(HasValue1 != HasValue2);
    cast<StructValue>(&MergedVal)->setProperty("has_value", HasValueTop);
    return true;
  }

  BoolValue &HasValueTop;
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
        test::checkDataflow<OptionalIntAnalysis>(
            Code, "target",
            [this](ASTContext &Context, Environment &Env) {
              assert(HasValueTop == nullptr);
              HasValueTop = &Env.takeOwnership(std::make_unique<BoolValue>());
              return OptionalIntAnalysis(Context, *HasValueTop);
            },
            [&Match](
                llvm::ArrayRef<
                    std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                    Results,
                ASTContext &ASTCtx) { Match(Results, ASTCtx); },
            {"-fsyntax-only", "-std=c++17"}, FilesContents),
        llvm::Succeeded());
  }

  BoolValue *HasValueTop = nullptr;
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
      [this](llvm::ArrayRef<
                 std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                 Results,
             ASTContext &ASTCtx) {
        ASSERT_THAT(Results,
                    ElementsAre(Pair("p3", _), Pair("p2", _), Pair("p1", _)));
        const Environment &Env1 = Results[2].second.Env;
        const Environment &Env2 = Results[1].second.Env;
        const Environment &Env3 = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return cast<StructValue>(Env.getValue(*FooDecl, SkipPast::None));
        };

        EXPECT_EQ(GetFooValue(Env1)->getProperty("has_value"),
                  &Env1.getBoolLiteralValue(false));
        EXPECT_EQ(GetFooValue(Env2)->getProperty("has_value"),
                  &Env2.getBoolLiteralValue(true));
        EXPECT_EQ(GetFooValue(Env3)->getProperty("has_value"), HasValueTop);
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p4", _), Pair("p3", _),
                                         Pair("p2", _), Pair("p1", _)));
        const Environment &Env1 = Results[3].second.Env;
        const Environment &Env2 = Results[2].second.Env;
        const Environment &Env3 = Results[1].second.Env;
        const Environment &Env4 = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        auto GetFooValue = [FooDecl](const Environment &Env) {
          return cast<StructValue>(Env.getValue(*FooDecl, SkipPast::None));
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *FooLoc = cast<ScalarStorageLocation>(
                    Env.getStorageLocation(*FooDecl, SkipPast::None));
                const auto *BarVal =
                    cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const auto *FooVal =
                    cast<StructValue>(Env.getValue(*FooDecl, SkipPast::None));
                EXPECT_EQ(FooVal->getProperty("has_value"),
                          &Env.getBoolLiteralValue(true));
              });
}

} // namespace
