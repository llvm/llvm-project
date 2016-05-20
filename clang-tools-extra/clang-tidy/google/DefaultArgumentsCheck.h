//===--- DefaultArgumentsCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_DEFAULT_ARGUMENTS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_DEFAULT_ARGUMENTS_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {

/// Checks that default parameters are not given for virtual methods.
///
/// See https://google.github.io/styleguide/cppguide.html#Default_Arguments
class DefaultArgumentsCheck : public ClangTidyCheck {
public:
  DefaultArgumentsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_DEFAULT_ARGUMENTS_H
