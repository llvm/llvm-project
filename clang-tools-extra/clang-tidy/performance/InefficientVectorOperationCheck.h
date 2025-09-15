//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENT_VECTOR_OPERATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENT_VECTOR_OPERATION_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// Finds possible inefficient `std::vector` operations (e.g. `push_back`) in
/// for loops that may cause unnecessary memory reallocations.
///
/// When EnableProto, also finds calls that add element to protobuf repeated
/// field without calling Reserve() first.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/inefficient-vector-operation.html
class InefficientVectorOperationCheck : public ClangTidyCheck {
public:
  InefficientVectorOperationCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  void addMatcher(const ast_matchers::DeclarationMatcher &TargetRecordDecl,
                  StringRef VarDeclName, StringRef VarDeclStmtName,
                  const ast_matchers::DeclarationMatcher &AppendMethodDecl,
                  StringRef AppendCallName, ast_matchers::MatchFinder *Finder);
  const std::vector<StringRef> VectorLikeClasses;

  // If true, also check inefficient operations for proto repeated fields.
  bool EnableProto;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENT_VECTOR_OPERATION_H
