//===--- DefinitionsInHeadersCheck.h - clang-tidy----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_DEFINITIONS_IN_HEADERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_DEFINITIONS_IN_HEADERS_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

// Finds non-extern non-inline function and variable definitions in header
// files, which can lead to potential ODR violations.
//
// There is one option:
// - `UseHeaderFileExtension`: Whether to use file extension (h, hh, hpp, hxx)
//   to distinguish header files. True by default.
//
// For the user-facing documentation see:
// http://clang.llvm.org/extra/clang-tidy/checks/misc-definitions-in-headers.html
class DefinitionsInHeadersCheck : public ClangTidyCheck {
public:
  DefinitionsInHeadersCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool UseHeaderFileExtension;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_DEFINITIONS_IN_HEADERS_H
