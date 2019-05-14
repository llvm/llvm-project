//===--- NonCopyableObjects.h - clang-tidy-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONCOPYABLEOBJECTS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONCOPYABLEOBJECTS_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace misc {

/// The check flags dereferences and non-pointer declarations of objects that
/// are not meant to be passed by value, such as C FILE objects.
class NonCopyableObjectsCheck : public ClangTidyCheck {
public:
  NonCopyableObjectsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONCOPYABLEOBJECTS_H
