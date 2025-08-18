//===--- ZirconTidyModule.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "TemporaryObjectsCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy {
namespace zircon {

/// This module is for Zircon-specific checks.
class ZirconModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<TemporaryObjectsCheck>(
        "zircon-temporary-objects");
  }
};

// Register the ZirconTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<ZirconModule>
    X("zircon-module", "Adds Zircon kernel checks.");
} // namespace zircon

// This anchor is used to force the linker to link in the generated object file
// and thus register the ZirconModule.
volatile int ZirconModuleAnchorSource = 0; // NOLINT(misc-use-internal-linkage)

} // namespace clang::tidy
