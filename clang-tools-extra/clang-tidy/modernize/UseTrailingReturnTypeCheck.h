//===--- UseTrailingReturnTypeCheck.h - clang-tidy---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETRAILINGRETURNTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETRAILINGRETURNTYPECHECK_H

#include "../ClangTidyCheck.h"
#include "clang/Lex/Token.h"

namespace clang {
namespace tidy {
namespace modernize {

struct ClassifiedToken {
  Token T;
  bool isQualifier;
  bool isSpecifier;
};

/// Rewrites function signatures to use a trailing return type.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-trailing-type-return.html
class UseTrailingReturnTypeCheck : public ClangTidyCheck {
public:
  UseTrailingReturnTypeCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  Preprocessor *PP = nullptr;

  SourceLocation findTrailingReturnTypeSourceLocation(
      const FunctionDecl &F, const FunctionTypeLoc &FTL, const ASTContext &Ctx,
      const SourceManager &SM, const LangOptions &LangOpts);
  llvm::Optional<SmallVector<ClassifiedToken, 8>>
  classifyTokensBeforeFunctionName(const FunctionDecl &F, const ASTContext &Ctx,
                                   const SourceManager &SM,
                                   const LangOptions &LangOpts);
  SourceRange findReturnTypeAndCVSourceRange(const FunctionDecl &F,
                                             const TypeLoc &ReturnLoc,
                                             const ASTContext &Ctx,
                                             const SourceManager &SM,
                                             const LangOptions &LangOpts);
  void keepSpecifiers(std::string &ReturnType, std::string &Auto,
                      SourceRange ReturnTypeCVRange, const FunctionDecl &F,
                      const FriendDecl *Fr, const ASTContext &Ctx,
                      const SourceManager &SM, const LangOptions &LangOpts);
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETRAILINGRETURNTYPECHECK_H
