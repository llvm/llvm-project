//===-- CheckNullabilityState.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckNullabilityState.h"
#include "../../TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullability.h"
#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullabilityAnalysis.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <utility>

namespace clang::dataflow::nullability {

NullabilityKind evaluateNullabilityAtCheck(llvm::StringRef Code) {
  NullabilityKind Result = NullabilityKind::Unspecified;
  llvm::Error Err = dataflow::test::checkDataflow<PointerNullabilityAnalysis>(
      dataflow::test::AnalysisInputs<PointerNullabilityAnalysis>(
          Code, ast_matchers::hasName("target"),
          [](ASTContext &Ctx, Environment &Env) {
            return PointerNullabilityAnalysis(Ctx, Env);
          })
          .withASTBuildArgs({"-fno-delayed-template-parsing", "-std=c++17"}),
      [&Result](const llvm::StringMap<
                    DataflowAnalysisState<PointerNullabilityLattice>> &Results,
                const dataflow::test::AnalysisOutputs &AO) {
        const Environment &Env =
            dataflow::test::getEnvironmentAtAnnotation(Results, "check");
        Result = getNullability(dataflow::test::getValueForDecl<PointerValue>(
                                    AO.ASTCtx, Env, "target_p"),
                                Env);
      });
  EXPECT_THAT_ERROR(std::move(Err), llvm::Succeeded());
  return Result;
}

} // namespace clang::dataflow::nullability
