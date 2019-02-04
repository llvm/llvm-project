//===--- PortabilityTidyModule.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "SIMDIntrinsicsCheck.h"

namespace clang {
namespace tidy {
namespace portability {

class PortabilityModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<SIMDIntrinsicsCheck>(
        "portability-simd-intrinsics");
  }
};

// Register the PortabilityModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<PortabilityModule>
    X("portability-module", "Adds portability-related checks.");

} // namespace portability

// This anchor is used to force the linker to link in the generated object file
// and thus register the PortabilityModule.
volatile int PortabilityModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
