//===--- QueryCheck.h - clang-tidy ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUSTOM_QUERYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUSTOM_QUERYCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace clang::tidy::custom {

/// FIXME: Write a short description.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/custom/query.html
class QueryCheck : public ClangTidyCheck {
public:
  QueryCheck(llvm::StringRef Name, const ClangTidyOptions::CustomCheckValue &V,
             ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  llvm::SmallVector<ast_matchers::dynamic::DynTypedMatcher> Matchers{};
  struct Diag {
    std::string Message;
    DiagnosticIDs::Level Level;
  };
  llvm::StringMap<llvm::SmallVector<Diag>> Diags{};
};

} // namespace clang::tidy::custom

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUSTOM_QUERYCHECK_H
