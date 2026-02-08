//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantQualifiedAliasCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include <cassert>
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

struct TypeLocInfo {
  TypeLoc Loc;
  bool HasQualifier = false;
};

} // namespace

static bool hasMacroInRange(SourceRange Range, const SourceManager &SM,
                            const LangOptions &LangOpts) {
  if (Range.isInvalid())
    return true;
  return utils::lexer::rangeContainsExpansionsOrDirectives(Range, SM, LangOpts);
}

static std::optional<TypeLocInfo> getTypeLocInfo(TypeLoc TL) {
  if (TL.isNull())
    return std::nullopt;

  const auto MakeTypeLocInfo = [](auto TypeTL) {
    const bool HasQualifier =
        static_cast<bool>(TypeTL.getQualifierLoc().getNestedNameSpecifier());
    return TypeLocInfo{TypeTL, HasQualifier};
  };

  if (const auto TypedefTL = TL.getAs<TypedefTypeLoc>())
    return MakeTypeLocInfo(TypedefTL);

  if (const auto TagTL = TL.getAs<TagTypeLoc>())
    return MakeTypeLocInfo(TagTL);

  return std::nullopt;
}

static const NamedDecl *getNamedDeclFromTypeLoc(TypeLoc TL) {
  if (const auto TypedefTL = TL.getAs<TypedefTypeLoc>())
    return TypedefTL.getDecl();
  if (const auto TagTL = TL.getAs<TagTypeLoc>())
    return TagTL.getDecl();
  return nullptr;
}

static bool hasSameUnqualifiedName(const NamedDecl *LHS, const NamedDecl *RHS) {
  return LHS->getName() == RHS->getName();
}

static bool isNamespaceLikeDeclContext(const DeclContext *DC) {
  return isa<TranslationUnitDecl, NamespaceDecl>(DC);
}

static bool canUseUsingDeclarationForTarget(const TypeAliasDecl *Alias,
                                            const NamedDecl *Target) {
  const DeclContext *AliasContext = Alias->getDeclContext()->getRedeclContext();
  const DeclContext *TargetContext =
      Target->getDeclContext()->getRedeclContext();

  const auto *AliasRecord = dyn_cast<CXXRecordDecl>(AliasContext);
  if (!AliasRecord)
    return isNamespaceLikeDeclContext(TargetContext);

  const auto *TargetRecord = dyn_cast<CXXRecordDecl>(TargetContext);
  return TargetRecord && AliasRecord->isDerivedFrom(TargetRecord);
}

static bool hasTrailingSyntaxAfterRhsType(TypeLoc TL, const SourceManager &SM,
                                          const LangOptions &LangOpts) {
  const SourceLocation TypeEndLoc = TL.getEndLoc();
  if (TypeEndLoc.isInvalid() || TypeEndLoc.isMacroID())
    return true;
  const std::optional<Token> NextToken =
      utils::lexer::findNextTokenSkippingComments(TypeEndLoc, SM, LangOpts);
  return !NextToken || NextToken->isNot(tok::semi);
}

namespace {

AST_MATCHER(TypeAliasDecl, isAliasTemplate) {
  return Node.getDescribedAliasTemplate() != nullptr;
}

AST_MATCHER(NamedDecl, isInMacro) { return Node.getLocation().isMacroID(); }

AST_MATCHER(TypeAliasDecl, hasAliasAttributes) {
  if (Node.hasAttrs())
    return true;
  const TypeSourceInfo *TSI = Node.getTypeSourceInfo();
  if (!TSI)
    return false;
  for (TypeLoc CurTL = TSI->getTypeLoc(); !CurTL.isNull();
       CurTL = CurTL.getNextTypeLoc())
    if (CurTL.getAs<AttributedTypeLoc>())
      return true;
  return false;
}

AST_MATCHER(TypeLoc, isNonDependentTypeLoc) {
  return !Node.getType().isNull() && !Node.getType()->isDependentType();
}

AST_MATCHER(TypeLoc, isNonElaboratedNominalTypeLoc) {
  const auto IsNonElaboratedTypeLoc = [](auto TL) {
    return !TL.isNull() && !TL.getElaboratedKeywordLoc().isValid();
  };
  return IsNonElaboratedTypeLoc(Node.getAs<TypedefTypeLoc>()) ||
         IsNonElaboratedTypeLoc(Node.getAs<TagTypeLoc>());
}

AST_MATCHER(TypeLoc, isMacroFreeTypeLoc) {
  const ASTContext &Context = Finder->getASTContext();
  return !hasMacroInRange(Node.getSourceRange(), Context.getSourceManager(),
                          Context.getLangOpts());
}

AST_MATCHER(TypeLoc, hasNoTrailingSyntaxAfterTypeLoc) {
  const ASTContext &Context = Finder->getASTContext();
  return !hasTrailingSyntaxAfterRhsType(Node, Context.getSourceManager(),
                                        Context.getLangOpts());
}

AST_MATCHER(TypeAliasDecl, hasUsingDeclarationEquivalentTarget) {
  const TypeSourceInfo *TSI = Node.getTypeSourceInfo();
  if (!TSI)
    return false;
  const std::optional<TypeLocInfo> TypeInfo = getTypeLocInfo(TSI->getTypeLoc());
  if (!TypeInfo || !TypeInfo->HasQualifier)
    return false;
  const NamedDecl *Target = getNamedDeclFromTypeLoc(TypeInfo->Loc);
  return Target && hasSameUnqualifiedName(&Node, Target) &&
         canUseUsingDeclarationForTarget(&Node, Target);
}

} // namespace

RedundantQualifiedAliasCheck::RedundantQualifiedAliasCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      OnlyNamespaceScope(Options.get("OnlyNamespaceScope", false)) {}

void RedundantQualifiedAliasCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "OnlyNamespaceScope", OnlyNamespaceScope);
}

void RedundantQualifiedAliasCheck::registerMatchers(MatchFinder *Finder) {
  const auto ControlFlowInitStatementMatcher = stmt(
      anyOf(mapAnyOf(ifStmt, switchStmt, cxxForRangeStmt)
                .with(hasInitStatement(stmt(equalsBoundNode("initDeclStmt")))),
            forStmt(hasLoopInit(stmt(equalsBoundNode("initDeclStmt"))))));

  const auto AliasPreconditions =
      allOf(unless(isInMacro()), unless(isAliasTemplate()),
            unless(hasAliasAttributes()));
  const auto InControlFlowInit =
      allOf(hasParent(declStmt().bind("initDeclStmt")),
            hasAncestor(ControlFlowInitStatementMatcher));
  const auto RewriteableTypeLoc =
      typeLoc(allOf(isNonDependentTypeLoc(), isNonElaboratedNominalTypeLoc(),
                    isMacroFreeTypeLoc(), hasNoTrailingSyntaxAfterTypeLoc()))
          .bind("loc");

  const auto RedundantQualifiedAliasMatcher = typeAliasDecl(
      AliasPreconditions, unless(InControlFlowInit),
      hasUsingDeclarationEquivalentTarget(), hasTypeLoc(RewriteableTypeLoc));

  if (OnlyNamespaceScope) {
    Finder->addMatcher(typeAliasDecl(RedundantQualifiedAliasMatcher,
                                     hasDeclContext(anyOf(translationUnitDecl(),
                                                          namespaceDecl())))
                           .bind("alias"),
                       this);
    return;
  }
  Finder->addMatcher(RedundantQualifiedAliasMatcher.bind("alias"), this);
}

void RedundantQualifiedAliasCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Alias = Result.Nodes.getNodeAs<TypeAliasDecl>("alias");
  assert(Alias && "matcher must bind alias");
  const auto *WrittenTLNode = Result.Nodes.getNodeAs<TypeLoc>("loc");
  assert(WrittenTLNode && "matcher must bind loc");
  const TypeLoc WrittenTL = *WrittenTLNode;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();

  const SourceLocation AliasLoc = Alias->getLocation();
  const SourceLocation RhsBeginLoc = WrittenTL.getBeginLoc();
  const CharSourceRange EqualRange = utils::lexer::findTokenTextInRange(
      CharSourceRange::getCharRange(AliasLoc, RhsBeginLoc), SM, LangOpts,
      [](const Token &Tok) { return Tok.is(tok::equal); });
  if (EqualRange.isInvalid())
    return;

  auto Diag = diag(Alias->getLocation(),
                   "type alias is redundant; use a using-declaration instead");

  Diag << FixItHint::CreateRemoval(Alias->getLocation())
       << FixItHint::CreateRemoval(EqualRange.getBegin());
}

} // namespace clang::tidy::readability
