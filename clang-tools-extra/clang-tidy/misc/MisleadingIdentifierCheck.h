//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISLEADINGIDENTIFIERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISLEADINGIDENTIFIERCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Finds identifiers that contain Unicode characters with right-to-left
/// direction, which can be confusing as they may change the understanding of a
/// whole statement line.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/misc/misleading-identifier.html
class MisleadingIdentifierCheck : public ClangTidyCheck {
public:
  MisleadingIdentifierCheck(StringRef Name, ClangTidyContext *Context);
  ~MisleadingIdentifierCheck() override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MISLEADINGIDENTIFIERCHECK_H
