//===--- Lifetime.h - clang-tidy --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIR_LIFETIME_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIR_LIFETIME_H

#include "../ClangTidyCheck.h"
#include <optional>

namespace clang::tidy::cir {

struct CIROpts {
  std::vector<StringRef> RemarksList;
  std::vector<StringRef> HistoryList;
  unsigned HistLimit;
};
class Lifetime : public ClangTidyCheck {
public:
  Lifetime(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void setupAndRunClangIRLifetimeChecker(ASTContext &astCtx);

  CodeGenOptions codeGenOpts;
  CIROpts cirOpts;
};

} // namespace clang::tidy::cir

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIR_LIFETIME_H
