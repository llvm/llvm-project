//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantQualifiedAliasCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

struct NominalTypeLocInfo {
  TypeLoc Loc;
  bool HasQualifier = false;
};

struct EqualTokenInfo {
  SourceLocation AfterEqualLoc;
  bool SawComment = false;
};

} // namespace

static bool hasMacroInRange(SourceRange Range, const SourceManager &SM,
                            const LangOptions &LangOpts) {
  if (Range.isInvalid())
    return true;
  return utils::lexer::rangeContainsExpansionsOrDirectives(Range, SM, LangOpts);
}

static std::optional<NominalTypeLocInfo> peelToNominalTypeLoc(TypeLoc TL) {
  while (!TL.isNull()) {
    if (const auto ParenTL = TL.getAs<ParenTypeLoc>()) {
      TL = ParenTL.getInnerLoc();
      continue;
    }
    if (const auto AttrTL = TL.getAs<AttributedTypeLoc>()) {
      // Preserve any attributes by rewriting only the alias name and '='.
      TL = AttrTL.getModifiedLoc();
      continue;
    }

    if (const auto TypedefTL = TL.getAs<TypedefTypeLoc>()) {
      // Avoid rewriting aliases that use an elaborated keyword
      // (class/struct/enum).
      if (TypedefTL.getElaboratedKeywordLoc().isValid())
        return std::nullopt;
      const bool HasQualifier = static_cast<bool>(
          TypedefTL.getQualifierLoc().getNestedNameSpecifier());
      return NominalTypeLocInfo{TypedefTL, HasQualifier};
    }

    if (const auto TagTL = TL.getAs<TagTypeLoc>()) {
      // Avoid rewriting aliases that use an elaborated keyword
      // (class/struct/enum).
      if (TagTL.getElaboratedKeywordLoc().isValid())
        return std::nullopt;
      const bool HasQualifier =
          static_cast<bool>(TagTL.getQualifierLoc().getNestedNameSpecifier());
      return NominalTypeLocInfo{TagTL, HasQualifier};
    }

    return std::nullopt;
  }
  return std::nullopt;
}

namespace {

AST_MATCHER(TypeAliasDecl, isAliasTemplate) {
  return Node.getDescribedAliasTemplate() != nullptr;
}

AST_MATCHER(TypeLoc, hasQualifiedNominalTypeLoc) {
  std::optional<NominalTypeLocInfo> Result = peelToNominalTypeLoc(Node);
  return Result && Result->HasQualifier;
}

} // namespace

static const NamedDecl *getNamedDeclFromNominalTypeLoc(TypeLoc TL) {
  if (const auto TypedefTL = TL.getAs<TypedefTypeLoc>())
    return TypedefTL.getTypePtr()->getDecl();
  if (const auto TagTL = TL.getAs<TagTypeLoc>())
    return TagTL.getDecl();
  return nullptr;
}

static bool hasSameUnqualifiedName(const TypeAliasDecl *Alias,
                                   const NamedDecl *Target) {
  return Alias->getName() == Target->getName();
}

static std::optional<EqualTokenInfo>
findEqualTokenAfter(SourceLocation StartLoc, SourceLocation LimitLoc,
                    const SourceManager &SM, const LangOptions &LangOpts) {
  if (StartLoc.isInvalid() || LimitLoc.isInvalid())
    return std::nullopt;
  if (StartLoc.isMacroID() || LimitLoc.isMacroID())
    return std::nullopt;
  if (!SM.isBeforeInTranslationUnit(StartLoc, LimitLoc))
    return std::nullopt;

  const SourceLocation SpellingStart = SM.getSpellingLoc(StartLoc);
  const SourceLocation SpellingLimit = SM.getSpellingLoc(LimitLoc);
  const FileID File = SM.getFileID(SpellingStart);
  if (File != SM.getFileID(SpellingLimit))
    return std::nullopt;

  const StringRef Buf = SM.getBufferData(File);
  const char *StartChar = SM.getCharacterData(SpellingStart);
  Lexer Lex(SpellingStart, LangOpts, StartChar, StartChar, Buf.end());
  Lex.SetCommentRetentionState(true);

  Token Tok;
  bool SawComment = false;
  do {
    Lex.LexFromRawLexer(Tok);
    if (Tok.is(tok::comment))
      SawComment = true;
    if (Tok.is(tok::equal)) {
      Token NextTok;
      Lex.LexFromRawLexer(NextTok);
      // Return the location *after* '=' so removal is token-preserving.
      return EqualTokenInfo{NextTok.getLocation(), SawComment};
    }
  } while (Tok.isNot(tok::eof) && Tok.getLocation() < SpellingLimit);

  return std::nullopt;
}

static std::optional<FixItHint> buildRemovalFixItAfterEqual(
    const TypeAliasDecl *Alias, SourceLocation AfterEqualLoc,
    const SourceManager &SM, const LangOptions &LangOpts) {
  SourceLocation AliasLoc = Alias->getLocation();
  if (AliasLoc.isInvalid() || AfterEqualLoc.isInvalid())
    return std::nullopt;
  if (AliasLoc.isMacroID() || AfterEqualLoc.isMacroID())
    return std::nullopt;

  AliasLoc = Lexer::GetBeginningOfToken(AliasLoc, SM, LangOpts);
  if (AliasLoc.isInvalid())
    return std::nullopt;

  const CharSourceRange RemovalRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(AliasLoc, AfterEqualLoc), SM, LangOpts);
  if (RemovalRange.isInvalid())
    return std::nullopt;

  if (hasMacroInRange(RemovalRange.getAsRange(), SM, LangOpts))
    // Avoid rewriting tokens that come from macro expansions.
    return std::nullopt;

  return FixItHint::CreateRemoval(RemovalRange);
}

RedundantQualifiedAliasCheck::RedundantQualifiedAliasCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      OnlyNamespaceScope(Options.get("OnlyNamespaceScope", false)) {}

void RedundantQualifiedAliasCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "OnlyNamespaceScope", OnlyNamespaceScope);
}

void RedundantQualifiedAliasCheck::registerMatchers(MatchFinder *Finder) {
  if (OnlyNamespaceScope) {
    Finder->addMatcher(
        typeAliasDecl(
            unless(isAliasTemplate()), unless(isImplicit()),
            unless(isExpansionInSystemHeader()),
            hasTypeLoc(hasQualifiedNominalTypeLoc()),
            hasDeclContext(anyOf(translationUnitDecl(), namespaceDecl())))
            .bind("alias"),
        this);
    return;
  }
  Finder->addMatcher(typeAliasDecl(unless(isAliasTemplate()),
                                   unless(isImplicit()),
                                   unless(isExpansionInSystemHeader()),
                                   hasTypeLoc(hasQualifiedNominalTypeLoc()))
                         .bind("alias"),
                     this);
}

void RedundantQualifiedAliasCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Alias = Result.Nodes.getNodeAs<TypeAliasDecl>("alias");
  if (!Alias)
    return;

  if (Alias->getLocation().isInvalid() || Alias->getLocation().isMacroID())
    return;

  const TypeSourceInfo *TSI = Alias->getTypeSourceInfo();
  if (!TSI)
    return;

  const TypeLoc WrittenTL = TSI->getTypeLoc();
  if (WrittenTL.isNull())
    return;

  if (WrittenTL.getType()->isDependentType())
    return;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();
  if (hasMacroInRange(WrittenTL.getSourceRange(), SM, LangOpts))
    return;

  const std::optional<NominalTypeLocInfo> NominalInfo =
      peelToNominalTypeLoc(WrittenTL);
  if (!NominalInfo)
    return;

  if (!NominalInfo->HasQualifier)
    // Unqualified RHS would not gain anything from a using-declaration.
    return;

  const TypeLoc NominalTL = NominalInfo->Loc;
  const NamedDecl *Target = getNamedDeclFromNominalTypeLoc(NominalTL);
  if (!Target)
    return;

  if (!hasSameUnqualifiedName(Alias, Target))
    return;

  const SourceLocation AliasLoc = Alias->getLocation();
  const SourceLocation RhsBeginLoc = WrittenTL.getBeginLoc();
  const CharSourceRange AliasToRhsRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(AliasLoc, RhsBeginLoc), SM, LangOpts);
  if (AliasToRhsRange.isInvalid())
    return;
  if (hasMacroInRange(AliasToRhsRange.getAsRange(), SM, LangOpts))
    return;

  std::optional<EqualTokenInfo> EqualInfo =
      findEqualTokenAfter(AliasLoc, RhsBeginLoc, SM, LangOpts);
  if (!EqualInfo || EqualInfo->AfterEqualLoc.isInvalid())
    return;

  auto Diag = diag(Alias->getLocation(),
                   "type alias is redundant; use a using-declaration instead");

  if (EqualInfo->SawComment) {
    // Suppress fix-it: avoid deleting comments between alias name and '='.
    return;
  }

  if (const std::optional<FixItHint> Fix = buildRemovalFixItAfterEqual(
          Alias, EqualInfo->AfterEqualLoc, SM, LangOpts)) {
    Diag << *Fix;
  }
}

} // namespace clang::tidy::readability
