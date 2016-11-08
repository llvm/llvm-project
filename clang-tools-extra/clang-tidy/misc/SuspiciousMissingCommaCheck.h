//===--- SuspiciousMissingCommaCheck.h - clang-tidy--------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SUSPICIOUS_MISSING_COMMA_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SUSPICIOUS_MISSING_COMMA_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// This check finds string literals which are probably concatenated
/// accidentally.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-suspicious-missing-comma.html
class SuspiciousMissingCommaCheck : public ClangTidyCheck {
public:
  SuspiciousMissingCommaCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  // Minimal size of a string literals array to be considered by the checker.
  const unsigned SizeThreshold;
  // Maximal threshold ratio of suspicious string literals to be considered.
  const double RatioThreshold;
  // Maximal number of concatenated tokens.
  const unsigned MaxConcatenatedTokens;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SUSPICIOUS_MISSING_COMMA_H
