//===--- CFProtectionOptions.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines constants for -fcf-protection and other related flags.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CFPROTECTIONOPTIONS_H
#define LLVM_CLANG_BASIC_CFPROTECTIONOPTIONS_H

#include "llvm/Support/ErrorHandling.h"

namespace clang {

enum class CFBranchLabelSchemeKind {
  Default,
#define CF_BRANCH_LABEL_SCHEME(Kind, FlagVal) Kind,
#include "clang/Basic/CFProtectionOptions.def"
};

static inline const char *
getCFBranchLabelSchemeFlagVal(const CFBranchLabelSchemeKind Scheme) {
#define CF_BRANCH_LABEL_SCHEME(Kind, FlagVal)                                  \
  if (Scheme == CFBranchLabelSchemeKind::Kind)                                 \
    return #FlagVal;
#include "clang/Basic/CFProtectionOptions.def"

  llvm::report_fatal_error("invalid scheme");
}

} // namespace clang

#endif // #ifndef LLVM_CLANG_BASIC_CFPROTECTIONOPTIONS_H
