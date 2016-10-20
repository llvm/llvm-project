//===--- RedundantMemberInitCheck.h - clang-tidy----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANT_MEMBER_INIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANT_MEMBER_INIT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Finds member initializations that are unnecessary because the same default
/// constructor would be called if they were not present.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-member-init.html
class RedundantMemberInitCheck : public ClangTidyCheck {
public:
  RedundantMemberInitCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANT_MEMBER_INIT_H
