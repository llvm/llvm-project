//===--- PerformanceTidyModule.cpp - flang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../FlangTidyModule.h"
#include "../FlangTidyModuleRegistry.h"
#include "IntegerPowerCheck.h"
#include "PureProcedureCheck.h"

namespace Fortran::tidy {
namespace performance {

class PerformanceTidyModule : public FlangTidyModule {
public:
  void addCheckFactories(FlangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<IntegerPowerCheck>(
        "performance-integer-power");
    CheckFactories.registerCheck<PureProcedureCheck>(
        "performance-pure-procedure");
  }
};

} // namespace performance

// Register the PerformanceTidyModule using this statically initialized
// variable.
static FlangTidyModuleRegistry::Add<performance::PerformanceTidyModule>
    X("performance-module", "Performance Tidy Module");

// This anchor is used to force the linker to link in the generated object file
// and thus register the PerformanceModule.
// NOLINTNEXTLINE
volatile int PerformanceModuleAnchorSource = 0;

} // namespace Fortran::tidy
