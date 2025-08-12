//===--- LLVMLibcTidyModule.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "CalleeNamespaceCheck.h"
#include "ImplementationInNamespaceCheck.h"
#include "InlineFunctionDeclCheck.h"
#include "RestrictSystemLibcHeadersCheck.h"

namespace clang::tidy {
namespace llvm_libc {

class LLVMLibcModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<CalleeNamespaceCheck>(
        "llvmlibc-callee-namespace");
    CheckFactories.registerCheck<ImplementationInNamespaceCheck>(
        "llvmlibc-implementation-in-namespace");
    CheckFactories.registerCheck<InlineFunctionDeclCheck>(
        "llvmlibc-inline-function-decl");
    CheckFactories.registerCheck<RestrictSystemLibcHeadersCheck>(
        "llvmlibc-restrict-system-libc-headers");
  }
};

// Register the LLVMLibcTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<LLVMLibcModule>
    X("llvmlibc-module", "Adds LLVM libc standards checks.");

} // namespace llvm_libc

// This anchor is used to force the linker to link in the generated object file
// and thus register the LLVMLibcModule.
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int LLVMLibcModuleAnchorSource = 0;

} // namespace clang::tidy
