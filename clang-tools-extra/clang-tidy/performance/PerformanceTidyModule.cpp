//===-- PerformanceTidyModule.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "AvoidEndlCheck.h"
#include "FasterStringFindCheck.h"
#include "ForRangeCopyCheck.h"
#include "ImplicitConversionInLoopCheck.h"
#include "InefficientAlgorithmCheck.h"
#include "InefficientStringConcatenationCheck.h"
#include "InefficientVectorOperationCheck.h"
#include "MoveConstArgCheck.h"
#include "MoveConstructorInitCheck.h"
#include "NoAutomaticMoveCheck.h"
#include "NoIntToPtrCheck.h"
#include "NoexceptDestructorCheck.h"
#include "NoexceptMoveConstructorCheck.h"
#include "NoexceptSwapCheck.h"
#include "TriviallyDestructibleCheck.h"
#include "TypePromotionInMathFnCheck.h"
#include "UnnecessaryCopyInitialization.h"
#include "UnnecessaryValueParamCheck.h"

namespace clang::tidy {
namespace performance {

class PerformanceModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<AvoidEndlCheck>("performance-avoid-endl");
    CheckFactories.registerCheck<FasterStringFindCheck>(
        "performance-faster-string-find");
    CheckFactories.registerCheck<ForRangeCopyCheck>(
        "performance-for-range-copy");
    CheckFactories.registerCheck<ImplicitConversionInLoopCheck>(
        "performance-implicit-conversion-in-loop");
    CheckFactories.registerCheck<InefficientAlgorithmCheck>(
        "performance-inefficient-algorithm");
    CheckFactories.registerCheck<InefficientStringConcatenationCheck>(
        "performance-inefficient-string-concatenation");
    CheckFactories.registerCheck<InefficientVectorOperationCheck>(
        "performance-inefficient-vector-operation");
    CheckFactories.registerCheck<MoveConstArgCheck>(
        "performance-move-const-arg");
    CheckFactories.registerCheck<MoveConstructorInitCheck>(
        "performance-move-constructor-init");
    CheckFactories.registerCheck<NoAutomaticMoveCheck>(
        "performance-no-automatic-move");
    CheckFactories.registerCheck<NoIntToPtrCheck>("performance-no-int-to-ptr");
    CheckFactories.registerCheck<NoexceptDestructorCheck>(
        "performance-noexcept-destructor");
    CheckFactories.registerCheck<NoexceptMoveConstructorCheck>(
        "performance-noexcept-move-constructor");
    CheckFactories.registerCheck<NoexceptSwapCheck>(
        "performance-noexcept-swap");
    CheckFactories.registerCheck<TriviallyDestructibleCheck>(
        "performance-trivially-destructible");
    CheckFactories.registerCheck<TypePromotionInMathFnCheck>(
        "performance-type-promotion-in-math-fn");
    CheckFactories.registerCheck<UnnecessaryCopyInitialization>(
        "performance-unnecessary-copy-initialization");
    CheckFactories.registerCheck<UnnecessaryValueParamCheck>(
        "performance-unnecessary-value-param");
  }
};

// Register the PerformanceModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<PerformanceModule>
    X("performance-module", "Adds performance checks.");

} // namespace performance

// This anchor is used to force the linker to link in the generated object file
// and thus register the PerformanceModule.
volatile int PerformanceModuleAnchorSource = 0;

} // namespace clang::tidy
