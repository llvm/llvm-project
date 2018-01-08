//===--- CloexecDupCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_DUP_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_DUP_H

#include "CloexecCheck.h"

namespace clang {
namespace tidy {
namespace android {

/// dup() is better to be replaced by fcntl(), which has close-on-exec flag.
/// Find the usage of dup() and redirect user to use fcntl().
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-dup.html
class CloexecDupCheck : public CloexecCheck {
public:
  CloexecDupCheck(StringRef Name, ClangTidyContext *Context)
      : CloexecCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace android
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_DUP_H
