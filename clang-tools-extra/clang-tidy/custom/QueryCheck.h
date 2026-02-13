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
#include "clang/Basic/DiagnosticIDs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace clang::tidy::custom {

/// Implement of Clang-Query based check.
/// Not directly visible to users.
class QueryCheck : public ClangTidyCheck {
public:
  QueryCheck(llvm::StringRef Name, const ClangTidyOptions::CustomCheckValue &V,
             ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  llvm::SmallVector<ast_matchers::dynamic::DynTypedMatcher> Matchers;
  using BindNameMapToDiagMessage =
      llvm::StringMap<llvm::SmallVector<std::string>>;
  using DiagMaps =
      llvm::DenseMap<DiagnosticIDs::Level, BindNameMapToDiagMessage>;
  DiagMaps Diags;
};

} // namespace clang::tidy::custom

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CUSTOM_QUERYCHECK_H
