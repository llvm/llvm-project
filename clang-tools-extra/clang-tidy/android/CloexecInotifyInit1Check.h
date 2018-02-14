//===--- CloexecInotifyInit1Check.h - clang-tidy-----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_INOTIFY_INIT1_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_INOTIFY_INIT1_H

#include "CloexecCheck.h"

namespace clang {
namespace tidy {
namespace android {

/// Finds code that uses inotify_init1() without using the IN_CLOEXEC flag.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/android-cloexec-inotify-init1.html
class CloexecInotifyInit1Check : public CloexecCheck {
public:
  CloexecInotifyInit1Check(StringRef Name, ClangTidyContext *Context)
      : CloexecCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace android
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_INOTIFY_INIT1_H
