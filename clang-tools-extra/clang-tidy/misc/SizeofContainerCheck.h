//===--- SizeofContainerCheck.h - clang-tidy---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SIZEOF_CONTAINER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SIZEOF_CONTAINER_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Find usages of sizeof on expressions of STL container types. Most likely the
/// user wanted to use `.size()` instead.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-sizeof-container.html
class SizeofContainerCheck : public ClangTidyCheck {
public:
  SizeofContainerCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SIZEOF_CONTAINER_H
