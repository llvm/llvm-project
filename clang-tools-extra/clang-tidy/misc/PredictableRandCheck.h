//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_PREDICTABLERANDCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_PREDICTABLERANDCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Pseudorandom number generators are not genuinely random. The result of the
/// std::rand() function makes no guarantees as to the quality of the random
/// sequence produced.
/// This check warns for the usage of std::rand() function.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/misc/predictable-rand.html
class PredictableRandCheck : public ClangTidyCheck {
public:
  PredictableRandCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_PREDICTABLERANDCHECK_H
