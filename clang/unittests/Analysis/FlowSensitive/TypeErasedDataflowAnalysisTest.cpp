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
#include "clang/Analysis/FlowSensitive/NoopAnalysis.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
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
  DataflowAnalysisContext DACtx(std::make_unique<WatchedLiteralsSolver>());
  Environment Env(DACtx);

  return runDataflowAnalysis(CFCtx, Analysis, Env);
}

TEST(DataflowAnalysisTest, NoopAnalysis) {
  auto BlockStates = llvm::cantFail(
      runAnalysis<NoopAnalysis>("void target() {}", [](ASTContext &C) {
        return NoopAnalysis(C, false);
      }));
  EXPECT_EQ(BlockStates.size(), 2u);
  EXPECT_TRUE(BlockStates[0].has_value());
  EXPECT_TRUE(BlockStates[1].has_value());
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

// Models an analysis that uses flow conditions.
class SpecialBoolAnalysis
    : public DataflowAnalysis<SpecialBoolAnalysis, NoopLattice> {
public:
  explicit SpecialBoolAnalysis(ASTContext &Context)
      : DataflowAnalysis<SpecialBoolAnalysis, NoopLattice>(Context) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const Stmt *S, NoopLattice &, Environment &Env) {
    auto SpecialBoolRecordDecl = recordDecl(hasName("SpecialBool"));
    auto HasSpecialBoolType = hasType(SpecialBoolRecordDecl);

    if (const auto *E = selectFirst<CXXConstructExpr>(
            "call", match(cxxConstructExpr(HasSpecialBoolType).bind("call"), *S,
                          getASTContext()))) {
      auto &ConstructorVal = *Env.createValue(E->getType());
      ConstructorVal.setProperty("is_set", Env.getBoolLiteralValue(false));
      Env.setValue(*Env.getStorageLocation(*E, SkipPast::None), ConstructorVal);
    } else if (const auto *E = selectFirst<CXXMemberCallExpr>(
                   "call", match(cxxMemberCallExpr(callee(cxxMethodDecl(ofClass(
                                                       SpecialBoolRecordDecl))))
                                     .bind("call"),
                                 *S, getASTContext()))) {
      auto *Object = E->getImplicitObjectArgument();
      assert(Object != nullptr);

      auto *ObjectLoc =
          Env.getStorageLocation(*Object, SkipPast::ReferenceThenPointer);
      assert(ObjectLoc != nullptr);

      auto &ConstructorVal = *Env.createValue(Object->getType());
      ConstructorVal.setProperty("is_set", Env.getBoolLiteralValue(true));
      Env.setValue(*ObjectLoc, ConstructorVal);
    }
  }

  bool compareEquivalent(QualType Type, const Value &Val1,
                         const Environment &Env1, const Value &Val2,
                         const Environment &Env2) final {
    const auto *Decl = Type->getAsCXXRecordDecl();
    if (Decl == nullptr || Decl->getIdentifier() == nullptr ||
        Decl->getName() != "SpecialBool")
      return false;

    auto *IsSet1 = cast_or_null<BoolValue>(Val1.getProperty("is_set"));
    if (IsSet1 == nullptr)
      return true;

    auto *IsSet2 = cast_or_null<BoolValue>(Val2.getProperty("is_set"));
    if (IsSet2 == nullptr)
      return false;

    return Env1.flowConditionImplies(*IsSet1) ==
           Env2.flowConditionImplies(*IsSet2);
  }

  // Always returns `true` to accept the `MergedVal`.
  bool merge(QualType Type, const Value &Val1, const Environment &Env1,
             const Value &Val2, const Environment &Env2, Value &MergedVal,
             Environment &MergedEnv) final {
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
    if (Env1.flowConditionImplies(*IsSet1) &&
        Env2.flowConditionImplies(*IsSet2))
      MergedEnv.addToFlowCondition(IsSet);

    return true;
  }
};

class JoinFlowConditionsTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    ASSERT_THAT_ERROR(
        test::checkDataflow<SpecialBoolAnalysis>(
            Code, "target",
            [](ASTContext &Context, Environment &Env) {
              return SpecialBoolAnalysis(Context);
            },
            [&Match](
                llvm::ArrayRef<
                    std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                    Results,
                ASTContext &ASTCtx) { Match(Results, ASTCtx); },
            {"-fsyntax-only", "-std=c++17"}),
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
          return cast<BoolValue>(
              Env.getValue(*FooDecl, SkipPast::None)->getProperty("is_set"));
        };

        EXPECT_FALSE(Env1.flowConditionImplies(*GetFooValue(Env1)));
        EXPECT_TRUE(Env2.flowConditionImplies(*GetFooValue(Env2)));
        EXPECT_TRUE(Env3.flowConditionImplies(*GetFooValue(Env3)));
        EXPECT_TRUE(Env4.flowConditionImplies(*GetFooValue(Env3)));
      });
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
      auto &ConstructorVal = *Env.createValue(E->getType());
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

      auto &ConstructorVal = *Env.createValue(Object->getType());
      ConstructorVal.setProperty("has_value", Env.getBoolLiteralValue(true));
      Env.setValue(*ObjectLoc, ConstructorVal);
    }
  }

  bool compareEquivalent(QualType Type, const Value &Val1,
                         const Environment &Env1, const Value &Val2,
                         const Environment &Env2) final {
    // Nothing to say about a value that does not model an `OptionalInt`.
    if (!Type->isRecordType() ||
        Type->getAsCXXRecordDecl()->getQualifiedNameAsString() != "OptionalInt")
      return false;

    return Val1.getProperty("has_value") == Val2.getProperty("has_value");
  }

  bool merge(QualType Type, const Value &Val1, const Environment &Env1,
             const Value &Val2, const Environment &Env2, Value &MergedVal,
             Environment &MergedEnv) final {
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
      MergedVal.setProperty("has_value", HasValueTop);
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
              HasValueTop =
                  &Env.takeOwnership(std::make_unique<AtomicBoolValue>());
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
          return Env.getValue(*FooDecl, SkipPast::None);
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
  runDataflow(Code,
              [](llvm::ArrayRef<
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
                  return Env.getValue(*FooDecl, SkipPast::None);
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

                const auto *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                EXPECT_EQ(FooVal->getProperty("has_value"),
                          &Env.getBoolLiteralValue(true));
              });
}

class FlowConditionTest : public Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    ASSERT_THAT_ERROR(
        test::checkDataflow<NoopAnalysis>(
            Code, "target",
            [](ASTContext &Context, Environment &Env) {
              return NoopAnalysis(Context, true);
            },
            [&Match](
                llvm::ArrayRef<
                    std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                    Results,
                ASTContext &ASTCtx) { Match(Results, ASTCtx); },
            {"-fsyntax-only", "-std=c++17"}),
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                ASSERT_THAT(Results, ElementsAre(Pair("p2", _), Pair("p1", _)));

                const Environment &Env1 = Results[1].second.Env;
                auto *FooVal1 =
                    cast<BoolValue>(Env1.getValue(*FooDecl, SkipPast::None));
                EXPECT_TRUE(Env1.flowConditionImplies(*FooVal1));

                const Environment &Env2 = Results[0].second.Env;
                auto *FooVal2 =
                    cast<BoolValue>(Env2.getValue(*FooDecl, SkipPast::None));
                EXPECT_FALSE(Env2.flowConditionImplies(*FooVal2));
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                ASSERT_THAT(Results, ElementsAre(Pair("p2", _), Pair("p1", _)));

                const Environment &Env1 = Results[1].second.Env;
                auto *FooVal1 =
                    cast<BoolValue>(Env1.getValue(*FooDecl, SkipPast::None));
                EXPECT_FALSE(Env1.flowConditionImplies(*FooVal1));

                const Environment &Env2 = Results[0].second.Env;
                auto *FooVal2 =
                    cast<BoolValue>(Env2.getValue(*FooDecl, SkipPast::None));
                EXPECT_TRUE(Env2.flowConditionImplies(*FooVal2));
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        auto *FooVal = cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(*FooVal));
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                ASSERT_THAT(Results, ElementsAre(Pair("p2", _), Pair("p1", _)));

                const Environment &Env1 = Results[1].second.Env;
                auto *FooVal1 =
                    cast<BoolValue>(Env1.getValue(*FooDecl, SkipPast::None));
                auto *BarVal1 =
                    cast<BoolValue>(Env1.getValue(*BarDecl, SkipPast::None));
                EXPECT_TRUE(Env1.flowConditionImplies(*FooVal1));
                EXPECT_TRUE(Env1.flowConditionImplies(*BarVal1));

                const Environment &Env2 = Results[0].second.Env;
                auto *FooVal2 =
                    cast<BoolValue>(Env2.getValue(*FooDecl, SkipPast::None));
                auto *BarVal2 =
                    cast<BoolValue>(Env2.getValue(*BarDecl, SkipPast::None));
                EXPECT_FALSE(Env2.flowConditionImplies(*FooVal2));
                EXPECT_FALSE(Env2.flowConditionImplies(*BarVal2));
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                ASSERT_THAT(Results, ElementsAre(Pair("p2", _), Pair("p1", _)));

                const Environment &Env1 = Results[1].second.Env;
                auto *FooVal1 =
                    cast<BoolValue>(Env1.getValue(*FooDecl, SkipPast::None));
                auto *BarVal1 =
                    cast<BoolValue>(Env1.getValue(*BarDecl, SkipPast::None));
                EXPECT_FALSE(Env1.flowConditionImplies(*FooVal1));
                EXPECT_FALSE(Env1.flowConditionImplies(*BarVal1));

                const Environment &Env2 = Results[0].second.Env;
                auto *FooVal2 =
                    cast<BoolValue>(Env2.getValue(*FooDecl, SkipPast::None));
                auto *BarVal2 =
                    cast<BoolValue>(Env2.getValue(*BarDecl, SkipPast::None));
                EXPECT_FALSE(Env2.flowConditionImplies(*FooVal2));
                EXPECT_FALSE(Env2.flowConditionImplies(*BarVal2));
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                ASSERT_THAT(Results, ElementsAre(Pair("p2", _), Pair("p1", _)));

                const Environment &Env1 = Results[1].second.Env;
                auto *FooVal1 =
                    cast<BoolValue>(Env1.getValue(*FooDecl, SkipPast::None));
                auto *BarVal1 =
                    cast<BoolValue>(Env1.getValue(*BarDecl, SkipPast::None));
                EXPECT_FALSE(Env1.flowConditionImplies(*FooVal1));
                EXPECT_FALSE(Env1.flowConditionImplies(*BarVal1));

                const Environment &Env2 = Results[0].second.Env;
                auto *FooVal2 =
                    cast<BoolValue>(Env2.getValue(*FooDecl, SkipPast::None));
                auto *BarVal2 =
                    cast<BoolValue>(Env2.getValue(*BarDecl, SkipPast::None));
                EXPECT_TRUE(Env2.flowConditionImplies(*FooVal2));
                EXPECT_TRUE(Env2.flowConditionImplies(*BarVal2));
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
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                ASSERT_THAT(Results, ElementsAre(Pair("p2", _), Pair("p1", _)));

                const Environment &Env1 = Results[1].second.Env;
                auto *FooVal1 =
                    cast<BoolValue>(Env1.getValue(*FooDecl, SkipPast::None));
                auto *BarVal1 =
                    cast<BoolValue>(Env1.getValue(*BarDecl, SkipPast::None));
                EXPECT_TRUE(Env1.flowConditionImplies(*FooVal1));
                EXPECT_TRUE(Env1.flowConditionImplies(*BarVal1));

                const Environment &Env2 = Results[0].second.Env;
                auto *FooVal2 =
                    cast<BoolValue>(Env2.getValue(*FooDecl, SkipPast::None));
                auto *BarVal2 =
                    cast<BoolValue>(Env2.getValue(*BarDecl, SkipPast::None));
                EXPECT_FALSE(Env2.flowConditionImplies(*FooVal2));
                EXPECT_FALSE(Env2.flowConditionImplies(*BarVal2));
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Environment &Env = Results[0].second.Env;
        auto *FooVal = cast<BoolValue>(Env.getValue(*FooDecl, SkipPast::None));
        EXPECT_TRUE(Env.flowConditionImplies(*FooVal));
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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal =
            *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::Reference));

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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal =
            *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::Reference));

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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        auto &BarVal =
            *cast<BoolValue>(Env.getValue(*BarDecl, SkipPast::Reference));

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
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p2", _), Pair("p1", _)));
        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Environment &Env1 = Results[1].second.Env;
        auto &FooVal1 =
            *cast<BoolValue>(Env1.getValue(*FooDecl, SkipPast::Reference));
        EXPECT_TRUE(Env1.flowConditionImplies(FooVal1));

        const Environment &Env2 = Results[0].second.Env;
        auto &FooVal2 =
            *cast<BoolValue>(Env2.getValue(*FooDecl, SkipPast::Reference));
        EXPECT_FALSE(Env2.flowConditionImplies(FooVal2));
      });
}

} // namespace
