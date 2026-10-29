//===--- QualifiedNameRewrite.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QualifiedNameRewrite.h"

#include "AST.h"
#include "ParsedAST.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>

namespace clang {
namespace clangd {

const NamedDecl *canonicalDecl(const NamedDecl *D) {
  return D ? llvm::dyn_cast<NamedDecl>(D->getCanonicalDecl()) : nullptr;
}

bool isNamespaceQualifier(NestedNameSpecifierLoc Qualifier) {
  return Qualifier && Qualifier.getNestedNameSpecifier().getKind() ==
                          NestedNameSpecifier::Kind::Namespace;
}

const NamespaceDecl *getNamespaceFromQualifier(NestedNameSpecifier Qualifier) {
  if (!Qualifier || Qualifier.getKind() != NestedNameSpecifier::Kind::Namespace)
    return nullptr;
  return llvm::dyn_cast_or_null<NamespaceDecl>(
      Qualifier.getAsNamespaceAndPrefix().Namespace);
}

bool normalizeToMainFile(SourceLocation &Loc, const SourceManager &SM) {
  if (Loc.isMacroID()) {
    if (!SM.isMacroArgExpansion(Loc))
      return false;
    Loc = SM.getFileLoc(Loc);
  }
  return SM.getFileID(Loc) == SM.getMainFileID();
}

bool referenceMatchesTargets(const ReferenceLoc &Ref,
                             const llvm::DenseSet<const NamedDecl *> &Targets) {
  for (const auto *Target : Ref.Targets) {
    if (const auto *Canonical = canonicalDecl(Target);
        Canonical && Targets.contains(Canonical))
      return true;
  }
  return false;
}

SourceLocation getReferenceEnd(SourceLocation NameLoc, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  std::optional<Token> Tok = Lexer::findNextToken(NameLoc, SM, LangOpts);
  if (!Tok || Tok->isNot(tok::less))
    return Lexer::getLocForEndOfToken(NameLoc, 0, SM, LangOpts);

  unsigned Depth = 0;
  while (Tok) {
    switch (Tok->getKind()) {
    case tok::less:
      ++Depth;
      break;
    case tok::greater:
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      --Depth;
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      break;
    case tok::greatergreater:
      if (Depth <= 1)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      Depth -= 2;
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      break;
    default:
      if (Depth == 0)
        return Lexer::getLocForEndOfToken(Tok->getLocation(), 0, SM, LangOpts);
      break;
    }
    Tok = Lexer::findNextToken(Tok->getLocation(), SM, LangOpts);
  }
  return Lexer::getLocForEndOfToken(NameLoc, 0, SM, LangOpts);
}

void forEachQualifiedReferenceToRewrite(
    const Tweak::Selection &Inputs,
    const llvm::DenseSet<const NamedDecl *> &Targets,
    llvm::function_ref<void(const QualifiedReference &)> Callback) {
  auto &SM = Inputs.AST->getSourceManager();

  for (const auto *D : Inputs.AST->getLocalTopLevelDecls()) {
    findExplicitReferences(
        D,
        [&](ReferenceLoc Ref) {
          if (!Ref.Qualifier || Ref.Targets.empty() || Ref.IsDecl)
            return;
          if (!isNamespaceQualifier(Ref.Qualifier))
            return;
          if (!referenceMatchesTargets(Ref, Targets))
            return;

          QualifiedReference QRef;
          QRef.QualifierLoc = Ref.Qualifier.getBeginLoc();
          QRef.NameLoc = Ref.NameLoc;
          if (!normalizeToMainFile(QRef.QualifierLoc, SM) ||
              !normalizeToMainFile(QRef.NameLoc, SM))
            return;

          Callback(QRef);
        },
        Inputs.AST->getHeuristicResolver());
  }
}

namespace {

std::string getNNSLAsString(NestedNameSpecifierLoc NNSL,
                            const PrintingPolicy &Policy) {
  std::string Out;
  llvm::raw_string_ostream OutStream(Out);
  NNSL.getNestedNameSpecifier().print(OutStream, Policy);
  return OutStream.str();
}

PreparedQualifiedSelection
computePreparedQualifiedSelection(const Tweak::Selection &Inputs) {
  PreparedQualifiedSelection Out;
  const auto *Node = Inputs.ASTSelection.commonAncestor();
  if (!Node)
    return Out;

  for (const auto *N = Node; N; N = N->Parent) {
    if (const auto *TAD = N->ASTNode.get<TypeAliasDecl>()) {
      if (TAD->getIdentifier())
        Out.SelectedAliasDecl = TAD;
      break;
    }
    if (const auto *UD = N->ASTNode.get<UsingDecl>()) {
      if (UD->getIdentifier())
        Out.SelectedAliasDecl = UD;
      break;
    }
  }

  // Normalize the selection to the outermost AST node that still represents
  // the qualified spelling we want to rewrite.
  for (; Node->Parent; Node = Node->Parent) {
    if (Node->ASTNode.get<NestedNameSpecifierLoc>())
      continue;
    if (const auto *T = Node->ASTNode.get<TypeLoc>()) {
      if (Node->Parent->ASTNode.get<NestedNameSpecifierLoc>())
        continue;
      if (llvm::isa<TagType, TemplateSpecializationType, TypedefType, UsingType,
                    UnresolvedUsingType>(T->getTypePtr()))
        break;
      if (Node->Parent->ASTNode.get<TypeLoc>())
        continue;
    }
    break;
  }

  SourceRange SpelledNameRange;
  if (const auto *D = Node->ASTNode.get<DeclRefExpr>()) {
    if (D->getDecl()->getIdentifier()) {
      Out.QualifierToRemove = D->getQualifierLoc();
      SpelledNameRange = D->getSourceRange();
      if (auto AngleLoc = D->getLAngleLoc(); AngleLoc.isValid())
        SpelledNameRange.setEnd(AngleLoc.getLocWithOffset(-1));
      Out.MustInsertAfterLoc = D->getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(D->getDecl()))
        Out.Targets.insert(Canonical);
    }
  } else if (const auto *T = Node->ASTNode.get<TypeLoc>()) {
    switch (T->getTypeLocClass()) {
    case TypeLoc::TemplateSpecialization: {
      auto TL = T->castAs<TemplateSpecializationTypeLoc>();
      Out.QualifierToRemove = TL.getQualifierLoc();
      if (!Out.QualifierToRemove)
        break;
      SpelledNameRange = TL.getTemplateNameLoc();
      if (auto *TD = TL.getTypePtr()->getTemplateName().getAsTemplateDecl(
              /*IgnoreDeduced=*/true)) {
        Out.MustInsertAfterLoc = TD->getBeginLoc();
        if (const auto *Canonical = canonicalDecl(TD))
          Out.Targets.insert(Canonical);
      }
      break;
    }
    case TypeLoc::Enum:
    case TypeLoc::Record:
    case TypeLoc::InjectedClassName: {
      auto TL = T->castAs<TagTypeLoc>();
      Out.QualifierToRemove = TL.getQualifierLoc();
      if (!Out.QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      Out.MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        Out.Targets.insert(Canonical);
      break;
    }
    case TypeLoc::Typedef: {
      auto TL = T->castAs<TypedefTypeLoc>();
      Out.QualifierToRemove = TL.getQualifierLoc();
      if (!Out.QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      Out.MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        Out.Targets.insert(Canonical);
      break;
    }
    case TypeLoc::UnresolvedUsing: {
      auto TL = T->castAs<UnresolvedUsingTypeLoc>();
      Out.QualifierToRemove = TL.getQualifierLoc();
      if (!Out.QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      Out.MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        Out.Targets.insert(Canonical);
      break;
    }
    case TypeLoc::Using: {
      auto TL = T->castAs<UsingTypeLoc>();
      Out.QualifierToRemove = TL.getQualifierLoc();
      if (!Out.QualifierToRemove)
        break;
      SpelledNameRange = TL.getNameLoc();
      Out.MustInsertAfterLoc = TL.getDecl()->getBeginLoc();
      if (const auto *Canonical = canonicalDecl(TL.getDecl()))
        Out.Targets.insert(Canonical);
      break;
    }
    default:
      break;
    }
    if (Out.QualifierToRemove)
      SpelledNameRange.setBegin(Out.QualifierToRemove.getBeginLoc());
  }

  if (!Out.QualifierToRemove || Out.Targets.empty() ||
      !isNamespaceQualifier(Out.QualifierToRemove))
    return Out;

  auto &SM = Inputs.AST->getSourceManager();
  if (SM.isMacroBodyExpansion(Out.QualifierToRemove.getBeginLoc()) ||
      !SM.isWrittenInSameFile(Out.QualifierToRemove.getBeginLoc(),
                              Out.QualifierToRemove.getEndLoc()))
    return Out;

  const auto &TB = Inputs.AST->getTokens();
  auto SpelledTokens =
      TB.spelledForExpanded(TB.expandedTokens(SpelledNameRange));
  if (!SpelledTokens)
    return Out;
  auto SpelledRange =
      syntax::Token::range(SM, SpelledTokens->front(), SpelledTokens->back());
  std::tie(Out.SpelledQualifier, Out.SpelledName) =
      splitQualifiedName(SpelledRange.text(SM));
  Out.QualifierToSpell = getNNSLAsString(
      Out.QualifierToRemove, Inputs.AST->getASTContext().getPrintingPolicy());
  if (!llvm::StringRef(Out.QualifierToSpell).ends_with(Out.SpelledQualifier) ||
      Out.SpelledName.empty())
    return Out;

  Out.HasQualifiedRef = true;
  return Out;
}

struct CachedPreparedSelection {
  const Tweak::Selection *SelectionPtr = nullptr;
  const ParsedAST *AST = nullptr;
  const ASTContext *ASTCtx = nullptr;
  const char *CodeData = nullptr;
  unsigned SelectionBegin = 0;
  unsigned SelectionEnd = 0;
  const SelectionTree::Node *CommonAncestor = nullptr;
  unsigned ConsumedClientsMask = 0;
  PreparedQualifiedSelection Prepared;
};

} // namespace

const PreparedQualifiedSelection &
prepareQualifiedSelection(const Tweak::Selection &Inputs,
                          QualifiedRewriteClient Client) {
  static thread_local CachedPreparedSelection Cache;
  const auto ClientBit = static_cast<unsigned>(Client);
  const SelectionTree::Node *CommonAncestor =
      Inputs.ASTSelection.commonAncestor();
  const ASTContext *ASTCtx = &Inputs.AST->getASTContext();

  if (Cache.SelectionPtr == &Inputs && Cache.AST == Inputs.AST &&
      Cache.ASTCtx == ASTCtx && Cache.CodeData == Inputs.Code.data() &&
      Cache.SelectionBegin == Inputs.SelectionBegin &&
      Cache.SelectionEnd == Inputs.SelectionEnd &&
      Cache.CommonAncestor == CommonAncestor &&
      (Cache.ConsumedClientsMask & ClientBit) == 0) {
    Cache.ConsumedClientsMask |= ClientBit;
    return Cache.Prepared;
  }

  Cache.SelectionPtr = &Inputs;
  Cache.AST = Inputs.AST;
  Cache.ASTCtx = ASTCtx;
  Cache.CodeData = Inputs.Code.data();
  Cache.SelectionBegin = Inputs.SelectionBegin;
  Cache.SelectionEnd = Inputs.SelectionEnd;
  Cache.CommonAncestor = CommonAncestor;
  Cache.ConsumedClientsMask = ClientBit;
  Cache.Prepared = computePreparedQualifiedSelection(Inputs);
  return Cache.Prepared;
}

} // namespace clangd
} // namespace clang
