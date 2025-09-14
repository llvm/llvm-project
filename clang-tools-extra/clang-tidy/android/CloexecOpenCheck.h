//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_OPEN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_OPEN_H

#include "CloexecCheck.h"

namespace clang::tidy::android {

/// Finds code that opens file without using the O_CLOEXEC flag.
///
/// open(), openat(), and open64() had better to include O_CLOEXEC in their
/// flags argument. Only consider simple cases that the corresponding argument
/// is constant or binary operation OR among constants like 'O_CLOEXEC' or
/// 'O_CLOEXEC | O_RDONLY'. No constant propagation is performed.
///
/// Only the symbolic 'O_CLOEXEC' macro definition is checked, not the concrete
/// value.
class CloexecOpenCheck : public CloexecCheck {
public:
  CloexecOpenCheck(StringRef Name, ClangTidyContext *Context)
      : CloexecCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::android

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_OPEN_H
