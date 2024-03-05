//===--- IgnoredRemoveResultCheck.cpp - clang-tidy ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IgnoredRemoveResultCheck.h"

namespace clang::tidy::hicpp {

IgnoredRemoveResultCheck::IgnoredRemoveResultCheck(llvm::StringRef Name,
                                                   ClangTidyContext *Context)
    : UnusedReturnValueCheck(Name, Context,
                             {
                                 "::std::remove",
                                 "::std::remove_if",
                                 "::std::unique",
                             }) {
  // The constructor for ClangTidyCheck needs to have been called
  // before we can access options via Options.get().
  AllowCastToVoid = Options.get("AllowCastToVoid", true);
}

void IgnoredRemoveResultCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowCastToVoid", AllowCastToVoid);
}

} // namespace clang::tidy::hicpp
