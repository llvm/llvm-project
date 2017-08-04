//===--- AndroidTidyModule.cpp - clang-tidy--------------------------------===//
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
#include "CloexecCreatCheck.h"
#include "CloexecFopenCheck.h"
#include "CloexecOpenCheck.h"
#include "CloexecSocketCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

/// This module is for Android specific checks.
class AndroidModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<CloexecCreatCheck>("android-cloexec-creat");
    CheckFactories.registerCheck<CloexecFopenCheck>("android-cloexec-fopen");
    CheckFactories.registerCheck<CloexecOpenCheck>("android-cloexec-open");
    CheckFactories.registerCheck<CloexecSocketCheck>("android-cloexec-socket");
  }
};

// Register the AndroidTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<AndroidModule>
    X("android-module", "Adds Android platform checks.");

} // namespace android

// This anchor is used to force the linker to link in the generated object file
// and thus register the AndroidModule.
volatile int AndroidModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
