//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseUsingCheck.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
namespace {

AST_MATCHER(clang::LinkageSpecDecl, isExternCLinkage) {
  return Node.getLanguage() == clang::LinkageSpecLanguageIDs::C;
}

} // namespace

namespace clang::tidy::modernize {

static constexpr llvm::StringLiteral ExternCDeclName = "extern-c-decl";
static constexpr llvm::StringLiteral TypedefName = "typedef";

UseUsingCheck::UseUsingCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.get("IgnoreMacros", true)),
      IgnoreExternC(Options.get("IgnoreExternC", false)) {}

void UseUsingCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
  Options.store(Opts, "IgnoreExternC", IgnoreExternC);
}

void UseUsingCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      typedefDecl(
          unless(isInstantiated()),
          optionally(hasAncestor(
              linkageSpecDecl(isExternCLinkage()).bind(ExternCDeclName))))
          .bind(TypedefName),
      this);
}

void UseUsingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &MatchedDecl = *Result.Nodes.getNodeAs<TypedefDecl>(TypedefName);
  if (MatchedDecl.getLocation().isInvalid())
    return;

  const auto *ExternCDecl =
      Result.Nodes.getNodeAs<LinkageSpecDecl>(ExternCDeclName);
  if (ExternCDecl && IgnoreExternC)
    return;

  static constexpr llvm::StringLiteral UseUsingWarning =
      "use 'using' instead of 'typedef'";

  if (MatchedDecl.getBeginLoc().isMacroID()) {
    // Warn but do not fix if there is a macro.
    if (!IgnoreMacros)
      diag(MatchedDecl.getBeginLoc(), UseUsingWarning);
    return;
  }

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LO = getLangOpts();

  // typedefs with multiple comma-separated definitions produce multiple
  // consecutive TypedefDecl nodes whose SourceRanges overlap. Each range starts
  // at the "typedef" and then continues *across* previous definitions through
  // the end of the current TypedefDecl definition.
  const Token TokenBeforeName =
      *Lexer::findPreviousToken(MatchedDecl.getLocation(), SM, LO,
                                /*IncludeComments=*/false);
  const SourceRange RemovalRange = {
      TokenBeforeName.getEndLoc(),
      Lexer::getLocForEndOfToken(MatchedDecl.getLocation(), 1, SM, LO)};
  if (NextTypedefStartsANewSequence) {
    auto Diag = diag(MatchedDecl.getBeginLoc(), UseUsingWarning)
                << FixItHint::CreateInsertion(
                       MatchedDecl.getBeginLoc(),
                       ("using " + MatchedDecl.getName() + " =").str())
                << FixItHint::CreateRemoval(RemovalRange);

    SmallString<128> Scratch;
    for (const SourceLocation Loc :
         {MatchedDecl.getBeginLoc(), TokenBeforeName.getLocation()})
      if (Lexer::getSpelling(Loc, Scratch, SM, LO) == "typedef")
        Diag << FixItHint::CreateRemoval(Loc);

    FirstTypedefName = MatchedDecl.getName();
  } else {
    diag(LastCommaOrSemi, UseUsingWarning)
        << FixItHint::CreateReplacement(
               LastCommaOrSemi,
               (";\nusing " + MatchedDecl.getName() + " = " + FirstTypedefName)
                   .str())
        << FixItHint::CreateRemoval(RemovalRange);
  }

  const Token CommaOrSemi = *Lexer::findNextToken(
      MatchedDecl.getEndLoc(), SM, LO, /*IncludeComments=*/false);
  NextTypedefStartsANewSequence = CommaOrSemi.isNot(tok::TokenKind::comma);
  LastCommaOrSemi = CommaOrSemi.getLocation();
}

} // namespace clang::tidy::modernize
