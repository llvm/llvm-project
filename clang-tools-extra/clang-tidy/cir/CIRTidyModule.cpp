//===--- CIRTidyModule.cpp - clang-tidy -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "Lifetime.h"

namespace clang::tidy {
namespace cir {

class CIRModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<Lifetime>("cir-lifetime-check");
  }
};

} // namespace cir

// Register the CIRTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<cir::CIRModule>
    X("cir-module", "Adds ClangIR (CIR) based clang-tidy checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the CIRModule.
volatile int CIRModuleAnchorSource = 0;

} // namespace clang::tidy
