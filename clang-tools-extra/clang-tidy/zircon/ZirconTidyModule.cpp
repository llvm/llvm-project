//===--- ZirconTidyModule.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "TemporaryObjectsCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
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
volatile int ZirconModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
