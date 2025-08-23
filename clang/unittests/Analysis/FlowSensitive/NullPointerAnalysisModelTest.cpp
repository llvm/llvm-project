//===- NullPointerAnalysisModelTest.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a test for pointer nullability, specifically focused on
//  finding invalid dereferences, and unnecessary null-checks.
//  Only a limited set of operations are currently recognized. Notably, pointer
//  arithmetic, null-pointer assignments and _nullable/_nonnull attributes are
//  missing as of yet.
//
//  FIXME: Port over to the new type of dataflow test infrastructure
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/NullPointerAnalysisModel.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/ADT/StringMapEntry.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace clang::dataflow {
namespace {
using namespace ast_matchers;

constexpr char kVar[] = "var";
// constexpr char kKnown[] = "is-known";
constexpr char kIsNonnull[] = "is-nonnull";
constexpr char kIsNull[] = "is-null";

constexpr char kBoolTrue[] = "true";
constexpr char kBoolFalse[] = "false";
constexpr char kBoolInvalid[] = "invalid";
constexpr char kBoolUnknown[] = "unknown";
constexpr char kBoolNullptr[] = "is-nullptr";

std::string checkNullabilityState(BoolValue *value, const Environment &Env) {
  if (value == nullptr) {
    return std::string(kBoolNullptr);
  } else {
    int boolState = 0;
    if (Env.proves(value->formula())) {
      boolState |= 1;
    }
    if (Env.proves(Env.makeNot(*value).formula())) {
      boolState |= 2;
    }
    switch (boolState) {
    case 0:
      return kBoolUnknown;
    case 1:
      return kBoolTrue;
    case 2:
      return kBoolFalse;
    // If both the condition and its negation are satisfied, the program point
    // is proven to be impossible.
    case 3:
      return kBoolInvalid;
    default:
      llvm_unreachable("all cases covered in switch");
    }
  }
}

using namespace test;
using ::llvm::IsStringMapEntry;
using ::testing::AllOf;
using ::testing::Args;
using ::testing::IsSupersetOf;
using ::testing::NotNull;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P2(HasNullabilityState, null, nonnull,
           std::string("has nullability state where isNull is ") + null +
               " and isNonnull is " + nonnull) {
  return checkNullabilityState(
             cast_or_null<BoolValue>(arg.first->getProperty(kIsNonnull)),
             *arg.second) == nonnull &&
         checkNullabilityState(
             cast_or_null<BoolValue>(arg.first->getProperty(kIsNull)),
             *arg.second) == null;
}

MATCHER_P3(HoldsVariable, name, output, checks,
           ((negation ? "doesn't hold" : "holds") +
            llvm::StringRef(" a variable in its environment that ") +
            ::testing::DescribeMatcher<std::pair<Value *, Environment *>>(
                checks, negation))
               .str()) {
  auto MatchResults =
      match(functionDecl(hasDescendant(namedDecl(hasName(name)).bind(kVar))),
            *output.Target, output.ASTCtx);
  assert(!MatchResults.empty());

  const auto *pointerExpr = MatchResults[0].template getNodeAs<ValueDecl>(kVar);
  assert(pointerExpr != nullptr);

  const auto *ExprValue = arg.Env.getValue(*pointerExpr);

  if (ExprValue == nullptr) {
    return false;
  }

  return ExplainMatchResult(checks, std::pair{ExprValue, &arg.Env},
                            result_listener);
}

void RunDataflowAnalysis(
    llvm::StringRef Code,
    std::function<void(const llvm::StringMap<DataflowAnalysisState<
                           NullPointerAnalysisModel::Lattice>> &,
                       const AnalysisOutputs &)>
        VerifyResults) {
  ASSERT_THAT_ERROR(checkDataflow<NullPointerAnalysisModel>(
                        AnalysisInputs<NullPointerAnalysisModel>(
                            Code, hasName("fun"),
                            [](ASTContext &C, Environment &Env) {
                              return NullPointerAnalysisModel(C);
                            })
                            .withASTBuildArgs({"-fsyntax-only", "-std=c++17"}),
                        VerifyResults),
                    llvm::Succeeded());
}

template <typename MatcherFactory>
void ExpectDataflowResult(llvm::StringRef Code, MatcherFactory Expectations) {
  RunDataflowAnalysis(
      Code,
      /*VerifyResults=*/
      [&Expectations](const llvm::StringMap<DataflowAnalysisState<
                          NullPointerAnalysisModel::Lattice>> &Results,
                      const AnalysisOutputs &Output) {
        EXPECT_THAT(Results, Expectations(Output));
      });
}

TEST(NullCheckAfterDereferenceTest, Operations) {
  std::string Code = R"(
    struct S {
      int a;
    };

    void fun(int *Deref, S *Arrow) {
      *Deref = 0;
      Arrow->a = 20;
      // [[p]]
    }
  )";
  ExpectDataflowResult(Code, [](const AnalysisOutputs &Output) -> auto {
    return UnorderedElementsAre(IsStringMapEntry(
        "p", AllOf(HoldsVariable("Deref", Output,
                                 HasNullabilityState(kBoolFalse, kBoolTrue)),
                   HoldsVariable("Arrow", Output,
                                 HasNullabilityState(kBoolFalse, kBoolTrue)))));
  });
}

TEST(NullCheckAfterDereferenceTest, Conditional) {
  std::string Code = R"(
    void fun(int *p) {
      if (p) {
        (void)0; // [[p_true]]
      } else {
        (void)0; // [[p_false]]
      }

      (void)0; // [[p_merge]]
    }
  )";
  ExpectDataflowResult(Code, [](const AnalysisOutputs &Output) -> auto {
    return UnorderedElementsAre(
        IsStringMapEntry("p_true", HoldsVariable("p", Output,
                                                 HasNullabilityState(
                                                     kBoolFalse, kBoolTrue))),
        IsStringMapEntry("p_false", HoldsVariable("p", Output,
                                                  HasNullabilityState(
                                                      kBoolTrue, kBoolFalse))),
        IsStringMapEntry(
            "p_merge",
            HoldsVariable("p", Output,
                          HasNullabilityState(kBoolUnknown, kBoolUnknown))));
  });
}

TEST(NullCheckAfterDereferenceTest, ValueAssignment) {
  std::string Code = R"(
    using size_t = decltype(sizeof(void*));
    extern void *malloc(size_t);
    extern int *ext();

    void fun(int arg) {
      int *Addressof = &arg;
      int *Nullptr = nullptr;
      int *Nullptr2 = 0;

      int *MallocExpr = (int *)malloc(sizeof(int));
      int *NewExpr = new int(3);

      int *ExternalFn = ext();

      (void)0; // [[p]]
    }
  )";
  ExpectDataflowResult(Code, [](const AnalysisOutputs &Output) -> auto {
    return UnorderedElementsAre(IsStringMapEntry(
        "p",
        AllOf(HoldsVariable("Addressof", Output,
                            HasNullabilityState(kBoolFalse, kBoolTrue)),
              HoldsVariable("Nullptr", Output,
                            HasNullabilityState(kBoolTrue, kBoolFalse)),
              HoldsVariable("Nullptr", Output,
                            HasNullabilityState(kBoolTrue, kBoolFalse)),
              HoldsVariable("MallocExpr", Output,
                            HasNullabilityState(kBoolNullptr, kBoolNullptr)),
              // HoldsVariable("NewExpr", Output,
              // HasNullabilityState(kBoolFalse, kBoolTrue)),
              HoldsVariable("ExternalFn", Output,
                            HasNullabilityState(kBoolNullptr, kBoolNullptr)))));
  });
}

TEST(NullCheckAfterDereferenceTest, BooleanDependence) {
  std::string Code = R"(
    void fun(int *ptr, bool b) {
      if (b) {
        *ptr = 10;
      } else {
        ptr = nullptr;
      }
        
      (void)0; // [[p]]
    }
  )";
  RunDataflowAnalysis(Code, [](const llvm::StringMap<DataflowAnalysisState<
                                   NullPointerAnalysisModel::Lattice>> &Results,
                               const AnalysisOutputs &Outputs) {
    ASSERT_THAT(Results.keys(), UnorderedElementsAre("p"));
    const Environment &Env = getEnvironmentAtAnnotation(Results, "p");

    auto &BoolVal = getValueForDecl<BoolValue>(Outputs.ASTCtx, Env, "b");
    auto &PtrVal = getValueForDecl<PointerValue>(Outputs.ASTCtx, Env, "ptr");

    auto *IsNull = cast_or_null<BoolValue>(PtrVal.getProperty(kIsNull));
    auto *IsNonnull = cast_or_null<BoolValue>(PtrVal.getProperty(kIsNonnull));
    ASSERT_THAT(IsNull, NotNull());
    ASSERT_THAT(IsNonnull, NotNull());

    ASSERT_EQ(checkNullabilityState(
                  &Env.makeImplication(BoolVal, Env.makeNot(*IsNull)), Env),
              kBoolTrue);
    ASSERT_EQ(checkNullabilityState(
                  &Env.makeImplication(Env.makeNot(BoolVal), *IsNull), Env),
              kBoolTrue);
    ASSERT_EQ(
        checkNullabilityState(&Env.makeImplication(BoolVal, *IsNonnull), Env),
        kBoolTrue);
    ASSERT_EQ(
        checkNullabilityState(&Env.makeImplication(*IsNonnull, BoolVal), Env),
        kBoolTrue);
  });
}

} // namespace
} // namespace clang::dataflow
