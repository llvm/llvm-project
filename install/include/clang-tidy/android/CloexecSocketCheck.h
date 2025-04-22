//===--- CloexecSocketCheck.h - clang-tidy-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_SOCKET_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_SOCKET_H

#include "CloexecCheck.h"

namespace clang::tidy::android {

/// Finds code that uses socket() without using the SOCK_CLOEXEC flag.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/android/cloexec-socket.html
class CloexecSocketCheck : public CloexecCheck {
public:
  CloexecSocketCheck(StringRef Name, ClangTidyContext *Context)
      : CloexecCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::android

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_SOCKET_H
