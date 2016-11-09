//===--- NonCopyableObjects.h - clang-tidy-----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONCOPYABLEOBJECTS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NONCOPYABLEOBJECTS_H

#include "../ClangTidy.h"

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
