//===--- StringConstructorCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_CONSTRUCTOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_CONSTRUCTOR_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Finds suspicious string constructor and check their parameters.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-string-constructor.html
class StringConstructorCheck : public ClangTidyCheck {
public:
  StringConstructorCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool WarnOnLargeLength;
  const unsigned int LargeLengthThreshold;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_CONSTRUCTOR_H
