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
#include "CloexecAccept4Check.h"
#include "CloexecAcceptCheck.h"
#include "CloexecCreatCheck.h"
#include "CloexecDupCheck.h"
#include "CloexecEpollCreate1Check.h"
#include "CloexecEpollCreateCheck.h"
#include "CloexecFopenCheck.h"
#include "CloexecInotifyInit1Check.h"
#include "CloexecInotifyInitCheck.h"
#include "CloexecMemfdCreateCheck.h"
#include "CloexecOpenCheck.h"
#include "CloexecSocketCheck.h"
#include "ComparisonInTempFailureRetryCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

/// This module is for Android specific checks.
class AndroidModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<CloexecAccept4Check>("android-cloexec-accept4");
    CheckFactories.registerCheck<CloexecAcceptCheck>("android-cloexec-accept");
    CheckFactories.registerCheck<CloexecCreatCheck>("android-cloexec-creat");
    CheckFactories.registerCheck<CloexecEpollCreate1Check>(
        "android-cloexec-epoll-create1");
    CheckFactories.registerCheck<CloexecEpollCreateCheck>(
        "android-cloexec-epoll-create");
    CheckFactories.registerCheck<CloexecDupCheck>("android-cloexec-dup");
    CheckFactories.registerCheck<CloexecFopenCheck>("android-cloexec-fopen");
    CheckFactories.registerCheck<CloexecInotifyInitCheck>(
        "android-cloexec-inotify-init");
    CheckFactories.registerCheck<CloexecInotifyInit1Check>(
        "android-cloexec-inotify-init1");
    CheckFactories.registerCheck<CloexecMemfdCreateCheck>(
        "android-cloexec-memfd-create");
    CheckFactories.registerCheck<CloexecOpenCheck>("android-cloexec-open");
    CheckFactories.registerCheck<CloexecSocketCheck>("android-cloexec-socket");
    CheckFactories.registerCheck<ComparisonInTempFailureRetryCheck>(
        "android-comparison-in-temp-failure-retry");
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
