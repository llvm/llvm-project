//===--- MLIRTidyModule.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "OpBuilderCheck.h"

namespace clang::tidy {
namespace mlir_check {

class MLIRModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<OpBuilderCheck>("mlir-op-builder");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    return Options;
  }
};

// Register the ModuleModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<MLIRModule> X("mlir-module",
                                                  "Adds MLIR lint checks.");

} // namespace mlir_check

// This anchor is used to force the linker to link in the generated object file
// and thus register the MLIRModule.
volatile int MLIRModuleAnchorSource = 0; // NOLINT(misc-use-internal-linkage)

} // namespace clang::tidy
