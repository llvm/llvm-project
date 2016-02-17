//===--- MisplacedWideningCastCheck.h - clang-tidy---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISPLACED_WIDENING_CAST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISPLACED_WIDENING_CAST_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Find explicit redundant casts of calculation results to bigger type.
/// Typically from int to long. If the intention of the cast is to avoid loss
/// of precision then the cast is misplaced, and there can be loss of
/// precision. Otherwise such cast is ineffective.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-misplaced-widening-cast.html
class MisplacedWideningCastCheck : public ClangTidyCheck {
public:
  MisplacedWideningCastCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif
