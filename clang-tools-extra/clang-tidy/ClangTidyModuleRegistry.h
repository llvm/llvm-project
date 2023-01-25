//===--- ClangTidyModuleRegistry.h - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYMODULEREGISTRY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYMODULEREGISTRY_H

#include "ClangTidyModule.h"
#include "llvm/Support/Registry.h"

namespace clang::tidy {

typedef llvm::Registry<ClangTidyModule> ClangTidyModuleRegistry;

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYMODULEREGISTRY_H
