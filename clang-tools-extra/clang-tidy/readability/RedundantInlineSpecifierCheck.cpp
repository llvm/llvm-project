//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantInlineSpecifierCheck.h"
#include "../utils/FileExtensionsUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Token.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {
AST_POLYMORPHIC_MATCHER(isInlineSpecified,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        VarDecl)) {
  if (const auto *FD = dyn_cast<FunctionDecl>(&Node))
    return FD->isInlineSpecified();
  if (const auto *VD = dyn_cast<VarDecl>(&Node))
    return VD->isInlineSpecified();
  llvm_unreachable("Not a valid polymorphic type");
}

AST_POLYMORPHIC_MATCHER_P(isInternalLinkage,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                          VarDecl),
                          bool, StrictMode) {
  if (!StrictMode)
    return false;
  if (const auto *FD = dyn_cast<FunctionDecl>(&Node))
    return FD->getStorageClass() == SC_Static || FD->isInAnonymousNamespace();
  if (const auto *VD = dyn_cast<VarDecl>(&Node))
    return VD->isInAnonymousNamespace();
  llvm_unreachable("Not a valid polymorphic type");
}

/// Matches a non-member declaration that is spelled ``static`` and does not
/// live in a header file, where ``inline`` therefore buys nothing over plain
/// ``static``.
AST_MATCHER_P2(NamedDecl, isStaticInlineOutsideHeader, bool,
               DiagnoseStaticInline, FileExtensionsSet, HeaderFileExtensions) {
  // 'static' on a member means something unrelated to linkage.
  if (!DiagnoseStaticInline || Node.isCXXClassMember())
    return false;
  if (const auto *FD = dyn_cast<FunctionDecl>(&Node)) {
    if (FD->getStorageClass() != SC_Static)
      return false;
  } else if (const auto *VD = dyn_cast<VarDecl>(&Node)) {
    if (VD->getStorageClass() != SC_Static)
      return false;
  } else {
    return false;
  }
  return !utils::isPresumedLocInHeaderFile(
      Node.getLocation(), Finder->getASTContext().getSourceManager(),
      HeaderFileExtensions);
}
} // namespace

static SourceLocation getInlineTokenLocation(SourceRange RangeLocation,
                                             const SourceManager &Sources,
                                             const LangOptions &LangOpts) {
  const SourceLocation Loc = RangeLocation.getBegin();
  if (Loc.isMacroID())
    return {};

  Token FirstToken;
  Lexer::getRawToken(Loc, FirstToken, Sources, LangOpts, true);
  std::optional<Token> CurrentToken = FirstToken;
  while (CurrentToken && CurrentToken->getLocation() < RangeLocation.getEnd() &&
         CurrentToken->isNot(tok::eof)) {
    if (CurrentToken->is(tok::raw_identifier) &&
        CurrentToken->getRawIdentifier() == "inline")
      return CurrentToken->getLocation();

    CurrentToken = utils::lexer::findNextTokenSkippingComments(
        CurrentToken->getLocation(), Sources, LangOpts);
  }
  return {};
}

void RedundantInlineSpecifierCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "DiagnoseStaticInline", DiagnoseStaticInline);
}

void RedundantInlineSpecifierCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsPartOfRecordDecl = hasAncestor(recordDecl());
  const auto IsStaticInlineOutsideHeader =
      namedDecl(isStaticInlineOutsideHeader(DiagnoseStaticInline,
                                            getHeaderFileExtensions()))
          .bind("static_inline");

  Finder->addMatcher(
      functionDecl(isInlineSpecified(),
                   anyOf(isConstexpr(), isDeleted(),
                         allOf(isDefaulted(), IsPartOfRecordDecl),
                         isInternalLinkage(StrictMode),
                         allOf(isDefinition(), IsPartOfRecordDecl),
                         IsStaticInlineOutsideHeader))
          .bind("fun_decl"),
      this);

  if (StrictMode)
    Finder->addMatcher(
        functionTemplateDecl(
            has(functionDecl(allOf(isInlineSpecified(), isDefinition()))))
            .bind("templ_decl"),
        this);

  if (getLangOpts().CPlusPlus17) {
    Finder->addMatcher(
        varDecl(
            isInlineSpecified(),
            anyOf(allOf(isInternalLinkage(StrictMode),
                        unless(allOf(hasInitializer(expr()), IsPartOfRecordDecl,
                                     isStaticStorageClass()))),
                  allOf(isConstexpr(), IsPartOfRecordDecl),
                  IsStaticInlineOutsideHeader))
            .bind("var_decl"),
        this);
  }
}

template <typename T>
void RedundantInlineSpecifierCheck::handleMatchedDecl(
    const T *MatchedDecl, const SourceManager &Sources,
    const MatchFinder::MatchResult &Result, StringRef Message) {
  const SourceLocation Loc = getInlineTokenLocation(
      MatchedDecl->getSourceRange(), Sources, Result.Context->getLangOpts());
  if (Loc.isValid())
    diag(Loc, Message) << MatchedDecl << FixItHint::CreateRemoval(Loc);
}

void RedundantInlineSpecifierCheck::check(
    const MatchFinder::MatchResult &Result) {
  const SourceManager &Sources = *Result.SourceManager;
  const bool IsStaticInline =
      Result.Nodes.getNodeAs<Decl>("static_inline") != nullptr;

  if (const auto *MatchedDecl =
          Result.Nodes.getNodeAs<FunctionDecl>("fun_decl")) {
    handleMatchedDecl(MatchedDecl, Sources, Result,
                      IsStaticInline
                          ? "function %0 is declared 'static inline' outside "
                            "of a header; use 'static' instead"
                          : "function %0 has inline specifier but is "
                            "implicitly inlined");
  } else if (const auto *MatchedDecl =
                 Result.Nodes.getNodeAs<VarDecl>("var_decl")) {
    handleMatchedDecl(MatchedDecl, Sources, Result,
                      IsStaticInline
                          ? "variable %0 is declared 'static inline' outside "
                            "of a header; use 'static' instead"
                          : "variable %0 has inline specifier but is "
                            "implicitly inlined");
  } else if (const auto *MatchedDecl =
                 Result.Nodes.getNodeAs<FunctionTemplateDecl>("templ_decl")) {
    handleMatchedDecl(
        MatchedDecl, Sources, Result,
        "function %0 has inline specifier but is implicitly inlined");
  }
}

} // namespace clang::tidy::readability
