//===--- IgnoredRemoveResultCheck.h - clang-tidy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_IGNOREDREMOVERESULTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_IGNOREDREMOVERESULTCHECK_H

#include "../bugprone/UnusedReturnValueCheck.h"

namespace clang::tidy::hicpp {

/// Ensure that the result of std::remove, std::remove_if and std::unique
/// are not ignored according to rule 17.5.1.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/hicpp/ignored-remove-result.html
class IgnoredRemoveResultCheck : public bugprone::UnusedReturnValueCheck {
public:
  IgnoredRemoveResultCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
};

} // namespace clang::tidy::hicpp

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_IGNOREDREMOVERESULTCHECK_H
