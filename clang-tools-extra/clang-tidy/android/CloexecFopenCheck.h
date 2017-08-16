//===--- CloexecFopenCheck.h - clang-tidy------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source //
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_FOPEN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_FOPEN_H

#include "CloexecCheck.h"

namespace clang {
namespace tidy {
namespace android {

/// fopen() is suggested to include "e" in their mode string; like "re" would be
/// better than "r".
///
/// This check only works when corresponding argument is StringLiteral. No
/// constant propagation.
///
/// http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-fopen.html
class CloexecFopenCheck : public CloexecCheck {
public:
  CloexecFopenCheck(StringRef Name, ClangTidyContext *Context)
      : CloexecCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace android
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_FOPEN_H
