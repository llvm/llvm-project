//===--- MakeSmartPtrCheck.h - clang-tidy------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SMART_PTR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SMART_PTR_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
namespace tidy {
namespace modernize {

/// Base class for MakeSharedCheck and MakeUniqueCheck.
class MakeSmartPtrCheck : public ClangTidyCheck {
public:
  MakeSmartPtrCheck(StringRef Name, ClangTidyContext *Context,
                    StringRef MakeSmartPtrFunctionName);
  void registerMatchers(ast_matchers::MatchFinder *Finder) final;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) final;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

protected:
  using SmartPtrTypeMatcher = ast_matchers::internal::BindableMatcher<QualType>;

  /// Returns matcher that match with different smart pointer types.
  ///
  /// Requires to bind pointer type (qualType) with PointerType string declared
  /// in this class.
  virtual SmartPtrTypeMatcher getSmartPointerTypeMatcher() const = 0;

  /// Returns whether the C++ version is compatible with current check.
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

  static const char PointerType[];

private:
  utils::IncludeInserter Inserter;
  const std::string MakeSmartPtrFunctionHeader;
  const std::string MakeSmartPtrFunctionName;
  const bool IgnoreMacros;

  void checkConstruct(SourceManager &SM, ASTContext *Ctx,
                      const CXXConstructExpr *Construct, const QualType *Type,
                      const CXXNewExpr *New);
  void checkReset(SourceManager &SM, ASTContext *Ctx,
                  const CXXMemberCallExpr *Member, const CXXNewExpr *New);

  /// Returns true when the fixes for replacing CXXNewExpr are generated.
  bool replaceNew(DiagnosticBuilder &Diag, const CXXNewExpr *New,
                  SourceManager &SM, ASTContext *Ctx);
  void insertHeader(DiagnosticBuilder &Diag, FileID FD);
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SMART_PTR_H
