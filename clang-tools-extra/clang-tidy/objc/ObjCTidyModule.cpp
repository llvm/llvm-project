//===--- ObjCTidyModule.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "AvoidNSErrorInitCheck.h"
#include "AvoidSpinlockCheck.h"
#include "ForbiddenSubclassingCheck.h"
#include "PropertyDeclarationCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

class ObjCModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<AvoidNSErrorInitCheck>(
        "objc-avoid-nserror-init");
    CheckFactories.registerCheck<AvoidSpinlockCheck>(
        "objc-avoid-spinlock");
    CheckFactories.registerCheck<ForbiddenSubclassingCheck>(
        "objc-forbidden-subclassing");
    CheckFactories.registerCheck<PropertyDeclarationCheck>(
        "objc-property-declaration");
  }
};

// Register the ObjCTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<ObjCModule> X(
    "objc-module",
    "Adds Objective-C lint checks.");

} // namespace objc

// This anchor is used to force the linker to link in the generated object file
// and thus register the ObjCModule.
volatile int ObjCModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
