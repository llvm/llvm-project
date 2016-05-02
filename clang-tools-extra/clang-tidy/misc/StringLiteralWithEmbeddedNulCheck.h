//===--- StringLiteralWithEmbeddedNulCheck.h - clang-tidy--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_LITERAL_WITH_EMBEDDED_NUL_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_LITERAL_WITH_EMBEDDED_NUL_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Find suspicious string literals with embedded NUL characters.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-string-literal-with-embedded-nul.html
class StringLiteralWithEmbeddedNulCheck : public ClangTidyCheck {
public:
  StringLiteralWithEmbeddedNulCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_STRING_LITERAL_WITH_EMBEDDED_NUL_H
