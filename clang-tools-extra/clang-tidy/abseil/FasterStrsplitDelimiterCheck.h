//===--- FasterStrsplitDelimiterCheck.h - clang-tidy-------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_FASTERSTRSPLITDELIMITERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_FASTERSTRSPLITDELIMITERCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace abseil {

/// Finds instances of absl::StrSplit() or absl::MaxSplits() where the delimiter
/// is a single character string literal and replaces it with a character.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil-faster-strsplit-delimiter.html
class FasterStrsplitDelimiterCheck : public ClangTidyCheck {
public:
  FasterStrsplitDelimiterCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_FASTERSTRSPLITDELIMITERCHECK_H
