//===--- AddUsing.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Config.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace clang {
namespace clangd {
namespace {

// Tweak for removing full namespace qualifier under cursor on DeclRefExpr and
// types and adding "using" statement instead.
//
// Only qualifiers that refer exclusively to namespaces (no record types) are
// supported. There is some guessing of appropriate place to insert the using
// declaration. If we find any existing usings, we insert it there. If not, we
// insert right after the inner-most relevant namespace declaration. If there is
// none, or there is, but it was declared via macro, we insert above the first
// top level decl.
//
// This removes the qualifier from all occurrences of the symbol in the scope
// where the using declaration becomes visible.
class AddUsing : public Tweak {
public:
  const char *id() const override;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  // All of the following are set by prepare().
  // The qualifier to remove.
  NestedNameSpecifierLoc QualifierToRemove;
  // Qualified name to use when spelling the using declaration. This might be
  // different than SpelledQualifier in presence of error correction.
  std::string QualifierToSpell;
  // The name and qualifier as spelled in the code.
  llvm::StringRef SpelledQualifier;
  llvm::StringRef SpelledName;
  // If valid, the insertion point for "using" statement must come after this.
  // This is relevant when the type is defined in the main file, to make sure
  // the type/function is already defined at the point where "using" is added.
  SourceLocation MustInsertAfterLoc;
};
REGISTER_TWEAK(AddUsing)

std::string AddUsing::title() const {
  return std::string(llvm::formatv(
      "Add using-declaration for {0} and remove qualifier", SpelledName));
}

// Locates all "using" statements relevant to SelectionDeclContext.
class UsingFinder : public RecursiveASTVisitor<UsingFinder> {
public:
  UsingFinder(std::vector<const UsingDecl *> &Results,
              const DeclContext *SelectionDeclContext, const SourceManager &SM)
      : Results(Results), SelectionDeclContext(SelectionDeclContext), SM(SM) {}

  bool VisitUsingDecl(UsingDecl *D) {
    auto Loc = D->getUsingLoc();
    if (SM.getFileID(Loc) != SM.getMainFileID()) {
      return true;
    }
    if (D->getDeclContext()->Encloses(SelectionDeclContext)) {
      Results.push_back(D);
    }
    return true;
  }

  bool TraverseDecl(Decl *Node) {
    if (!Node)
      return true;
    // There is no need to go deeper into nodes that do not enclose selection,
    // since "using" there will not affect selection, nor would it make a good
    // insertion point.
    if (!Node->getDeclContext() ||
        Node->getDeclContext()->Encloses(SelectionDeclContext)) {
      return RecursiveASTVisitor<UsingFinder>::TraverseDecl(Node);
    }
    return true;
  }

private:
  std::vector<const UsingDecl *> &Results;
  const DeclContext *SelectionDeclContext;
  const SourceManager &SM;
};

const NamespaceDecl *getNamespaceFromQualifier(NestedNameSpecifier Qualifier) {
  if (!Qualifier || Qualifier.getKind() != NestedNameSpecifier::Kind::Namespace)
    return nullptr;
  return llvm::dyn_cast_or_null<NamespaceDecl>(
      Qualifier.getAsNamespaceAndPrefix().Namespace);
}

struct InsertionPointData {
  // Location to insert the "using" statement. If invalid then the statement
  // should not be inserted at all (it already exists).
  SourceLocation Loc;
  // Extra suffix to place after the "using" statement. Depending on what the
  // insertion point is anchored to, we may need one or more \n to ensure
  // proper formatting.
  std::string Suffix;
  // Whether using should be fully qualified, even if what the user typed was
  // not. This is based on our detection of the local style.
  bool AlwaysFullyQualify = false;
  // Scope in which the (inserted or already-existing) using-declaration is
  // visible, and the source location from which it takes effect.
  const DeclContext *EffectiveContext = nullptr;
  SourceLocation EffectiveLoc;
  // When anchored to a block-local UsingDecl, points to the enclosing Stmt
  // (e.g. CompoundStmt) in which that UsingDecl is visible.
  const Stmt *LocalScope = nullptr;
};

const Stmt *findLocalUsingScope(ASTContext &ASTCtx, const UsingDecl *U) {
  if (U->getDeclContext()->isFileContext())
    return nullptr;
  for (const auto &P : ASTCtx.getParents(*U))
    if (const auto *DS = P.get<DeclStmt>())
      for (const auto &SP : ASTCtx.getParents(*DS))
        if (const auto *S = SP.get<Stmt>())
          return S;
  return nullptr;
}

// Finds the best place to insert the "using" statement. Returns invalid
// SourceLocation if the "using" statement already exists.
//
// The insertion point might be a little awkward if the decl we're anchoring to
// has a comment in an unfortunate place (e.g. directly above function or using
// decl, or immediately following "namespace {". We should add some helpers for
// dealing with that and use them in other code modifications as well.
llvm::Expected<InsertionPointData>
findInsertionPoint(const Tweak::Selection &Inputs,
                   const NestedNameSpecifierLoc &QualifierToRemove,
                   const llvm::StringRef Name,
                   const SourceLocation MustInsertAfterLoc) {
  auto &SM = Inputs.AST->getSourceManager();
  const auto *TargetNS =
      getNamespaceFromQualifier(QualifierToRemove.getNestedNameSpecifier());
  if (!TargetNS)
    return error("Qualifier is not a namespace");
  const auto *CanonicalTargetNS = TargetNS->getCanonicalDecl();

  // Search for all using decls that affect this point in file. We need this for
  // two reasons: to skip adding "using" if one already exists and to find best
  // place to add it, if it doesn't exist.
  const UsingDecl *LastUsing = nullptr;
  std::vector<const UsingDecl *> Usings;
  UsingFinder(Usings, &Inputs.ASTSelection.commonAncestor()->getDeclContext(),
              SM)
      .TraverseAST(Inputs.AST->getASTContext());

  auto IsValidPoint = [&](const SourceLocation Loc) {
    return MustInsertAfterLoc.isInvalid() ||
           SM.isBeforeInTranslationUnit(MustInsertAfterLoc, Loc);
  };

  bool AlwaysFullyQualify = true;
  for (auto &U : Usings) {
    // Only "upgrade" to fully qualified is all relevant using decls are fully
    // qualified. Otherwise trust what the user typed.
    if (!U->getQualifier().isFullyQualified())
      AlwaysFullyQualify = false;

    if (SM.isBeforeInTranslationUnit(Inputs.Cursor, U->getUsingLoc()))
      // "Usings" is sorted, so we're done.
      break;
    if (const auto *Namespace = getNamespaceFromQualifier(U->getQualifier())) {
      if (Namespace->getCanonicalDecl() == CanonicalTargetNS &&
          U->getName() == Name) {
        InsertionPointData Out;
        Out.EffectiveContext = U->getDeclContext();
        Out.EffectiveLoc = U->getEndLoc();
        Out.LocalScope = findLocalUsingScope(Inputs.AST->getASTContext(), U);
        return Out;
      }
    }

    // Insertion point will be before last UsingDecl that affects cursor
    // position. For most cases this should stick with the local convention of
    // add using inside or outside namespace.
    LastUsing = U;
  }
  if (LastUsing && LastUsing->getUsingLoc().isValid() &&
      IsValidPoint(LastUsing->getUsingLoc())) {
    InsertionPointData Out;
    Out.Loc = LastUsing->getUsingLoc();
    Out.AlwaysFullyQualify = AlwaysFullyQualify;
    Out.EffectiveContext = LastUsing->getDeclContext();
    Out.EffectiveLoc = Out.Loc;
    Out.LocalScope =
        findLocalUsingScope(Inputs.AST->getASTContext(), LastUsing);
    return Out;
  }

  // No relevant "using" statements. Try the nearest namespace level.
  const DeclContext *ParentDeclCtx =
      &Inputs.ASTSelection.commonAncestor()->getDeclContext();
  while (ParentDeclCtx && !ParentDeclCtx->isFileContext()) {
    ParentDeclCtx = ParentDeclCtx->getLexicalParent();
  }
  if (auto *ND = llvm::dyn_cast_or_null<NamespaceDecl>(ParentDeclCtx)) {
    auto Toks = Inputs.AST->getTokens().expandedTokens(ND->getSourceRange());
    const auto *Tok = llvm::find_if(Toks, [](const syntax::Token &Tok) {
      return Tok.kind() == tok::l_brace;
    });
    if (Tok == Toks.end() || Tok->endLocation().isInvalid()) {
      return error("Namespace with no {{");
    }
    if (!Tok->endLocation().isMacroID() && IsValidPoint(Tok->endLocation())) {
      InsertionPointData Out;
      Out.Loc = Tok->endLocation();
      Out.Suffix = "\n";
      Out.EffectiveContext = ND;
      Out.EffectiveLoc = Out.Loc;
      return Out;
    }
  }
  // No using, no namespace, no idea where to insert. Try above the first
  // top level decl after MustInsertAfterLoc.
  auto TLDs = Inputs.AST->getLocalTopLevelDecls();
  for (const auto &TLD : TLDs) {
    if (!IsValidPoint(TLD->getBeginLoc()))
      continue;
    InsertionPointData Out;
    Out.Loc = SM.getExpansionLoc(TLD->getBeginLoc());
    Out.Suffix = "\n\n";
    Out.EffectiveContext = Inputs.AST->getASTContext().getTranslationUnitDecl();
    Out.EffectiveLoc = Out.Loc;
    return Out;
  }
  return error("Cannot find place to insert \"using\"");
}

bool isNamespaceForbidden(const Tweak::Selection &Inputs,
                          NestedNameSpecifier Namespace) {
  const auto *NS = getNamespaceFromQualifier(Namespace);
  if (!NS)
    return true;
  std::string NamespaceStr = printNamespaceScope(*NS);

  for (StringRef Banned : Config::current().Style.FullyQualifiedNamespaces) {
    StringRef PrefixMatch = NamespaceStr;
    if (PrefixMatch.consume_front(Banned) && PrefixMatch.consume_front("::"))
      return true;
  }

  return false;
}

std::string getNNSLAsString(NestedNameSpecifierLoc NNSL,
                            const PrintingPolicy &Policy) {
  std::string Out;
  llvm::raw_string_ostream OutStream(Out);
  NNSL.getNestedNameSpecifier().print(OutStream, Policy);
  return OutStream.str();
}

struct ExtractedQualifiedRef {
  NestedNameSpecifierLoc Qualifier;
  // Closed token range [QualifierBegin, NameToken].
  SourceRange SpelledNameRange;
  SourceLocation MustInsertAfterLoc;
};

std::optional<ExtractedQualifiedRef>
extractFromDeclRefExpr(const DeclRefExpr *D) {
  if (!D || !D->getDecl()->getIdentifier())
    return std::nullopt;
  NestedNameSpecifierLoc Qual = D->getQualifierLoc();
  if (!Qual)
    return std::nullopt;
  return ExtractedQualifiedRef{
      Qual, SourceRange(Qual.getBeginLoc(), D->getLocation()),
      D->getDecl()->getBeginLoc()};
}

std::optional<ExtractedQualifiedRef> extractFromTypeLoc(TypeLoc T) {
  if (!T)
    return std::nullopt;
  auto FromTemplateLoc = [](auto TL) -> std::optional<ExtractedQualifiedRef> {
    NestedNameSpecifierLoc Qual = TL.getQualifierLoc();
    SourceLocation NameLoc = TL.getTemplateNameLoc();
    if (!Qual || NameLoc.isInvalid())
      return std::nullopt;
    SourceLocation MustInsertAfter;
    if (auto *TD = TL.getTypePtr()->getTemplateName().getAsTemplateDecl(
            /*IgnoreDeduced=*/true))
      MustInsertAfter = TD->getBeginLoc();
    return ExtractedQualifiedRef{Qual, SourceRange(Qual.getBeginLoc(), NameLoc),
                                 MustInsertAfter};
  };
  auto FromDeclLoc = [](auto TL) -> std::optional<ExtractedQualifiedRef> {
    NestedNameSpecifierLoc Qual = TL.getQualifierLoc();
    SourceLocation NameLoc = TL.getNameLoc();
    if (!Qual || NameLoc.isInvalid())
      return std::nullopt;
    return ExtractedQualifiedRef{Qual, SourceRange(Qual.getBeginLoc(), NameLoc),
                                 TL.getDecl()->getBeginLoc()};
  };

  switch (T.getTypeLocClass()) {
  case TypeLoc::TemplateSpecialization:
    return FromTemplateLoc(T.castAs<TemplateSpecializationTypeLoc>());
  case TypeLoc::DeducedTemplateSpecialization:
    return FromTemplateLoc(T.castAs<DeducedTemplateSpecializationTypeLoc>());
  case TypeLoc::Enum:
  case TypeLoc::Record:
  case TypeLoc::InjectedClassName:
    return FromDeclLoc(T.castAs<TagTypeLoc>());
  case TypeLoc::Typedef:
    return FromDeclLoc(T.castAs<TypedefTypeLoc>());
  case TypeLoc::UnresolvedUsing:
    return FromDeclLoc(T.castAs<UnresolvedUsingTypeLoc>());
  case TypeLoc::Using:
    return FromDeclLoc(T.castAs<UsingTypeLoc>());
  default:
    return std::nullopt;
  }
}

// Returns true if `D` is declared inside `TargetNS` (unwrapping inline
// namespaces, unscoped enums, and UsingShadowDecls).
bool isFromTargetNamespace(const NamedDecl *D,
                           const NamespaceDecl *CanonicalTargetNS) {
  if (const auto *USD = llvm::dyn_cast<UsingShadowDecl>(D))
    D = USD->getTargetDecl();
  const DeclContext *DC = D->getDeclContext();
  while (DC && (DC->isInlineNamespace() || DC->isTransparentContext()))
    DC = DC->getParent();
  const auto *NS = llvm::dyn_cast_or_null<NamespaceDecl>(DC);
  return NS && NS->getCanonicalDecl() == CanonicalTargetNS;
}

// Collects replacements removing redundant namespace qualifiers from all
// visible, unshadowed references to `CanonicalTargetNS::TargetName` in the main
// file within `InsertionPoint.EffectiveContext`.
class QualifiedRefCollector
    : public RecursiveASTVisitor<QualifiedRefCollector> {
public:
  QualifiedRefCollector(ASTContext &ASTCtx, const syntax::TokenBuffer &TB,
                        const NamespaceDecl *CanonicalTargetNS,
                        llvm::StringRef TargetName,
                        const InsertionPointData &InsertionPoint,
                        tooling::Replacements &Repls)
      : SM(ASTCtx.getSourceManager()), TB(TB),
        CanonicalTargetNS(CanonicalTargetNS), TargetName(TargetName),
        TargetIdent(&ASTCtx.Idents.get(TargetName)),
        InsertionPoint(InsertionPoint), Repls(Repls),
        CurrentSemanticDC(InsertionPoint.LocalScope
                              ? InsertionPoint.EffectiveContext
                              : ASTCtx.getTranslationUnitDecl()),
        CurrentLexicalDC(CurrentSemanticDC) {}

  llvm::Error takeError() { return std::move(Err); }

  bool TraverseFriendDecl(FriendDecl *) { return true; }
  bool TraverseUsingDecl(UsingDecl *) { return true; }

  bool TraverseDecl(Decl *D) {
    if (!D)
      return true;

    const auto *DC = llvm::dyn_cast<DeclContext>(D);
    const DeclContext *SemDC = DC ? DC : D->getDeclContext();
    const DeclContext *LexDC = DC ? DC : D->getLexicalDeclContext();
    llvm::SaveAndRestore SaveSemantic(CurrentSemanticDC,
                                      SemDC ? SemDC : CurrentSemanticDC);
    llvm::SaveAndRestore SaveLexical(CurrentLexicalDC,
                                     LexDC ? LexDC : CurrentLexicalDC);
    const auto *DD = llvm::dyn_cast<DeclaratorDecl>(D);
    llvm::SaveAndRestore SaveOutOfLine(
        OutOfLineDecl, (DD && DD->getQualifierLoc()) ? DD : OutOfLineDecl);

    if ((!DC || DC->isTransparentContext()) &&
        !llvm::isa<RedeclarableTemplateDecl, ConceptDecl>(D))
      return RecursiveASTVisitor::TraverseDecl(D);

    size_t SavedLocals = LocalDecls.size();
    bool Res = RecursiveASTVisitor::TraverseDecl(D);
    LocalDecls.truncate(SavedLocals);
    return Res;
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) {
    if (!NNS)
      return true;
    // Skip the inner TypeLoc of a type-qualified specifier (e.g. `ns::Foo` in
    // `ns::Foo::bar()` or `void ns::Foo::bar() {}`), while still traversing any
    // template arguments inside it (e.g. `ns::Vec<ns::Foo>::type`).
    if (NNS.getNestedNameSpecifier().getKind() ==
        NestedNameSpecifier::Kind::Type)
      SkippedTypeLocStarts.insert(NNS.castAsTypeLoc().getBeginLoc());
    return RecursiveASTVisitor::TraverseNestedNameSpecifierLoc(NNS);
  }

  bool dataTraverseStmtPre(Stmt *S) {
    if (isScopedStmt(S))
      ScopeStack.push_back(LocalDecls.size());
    return true;
  }
  bool dataTraverseStmtPost(Stmt *S) {
    if (isScopedStmt(S))
      LocalDecls.truncate(ScopeStack.pop_back_val());
    return true;
  }

  bool TraverseLambdaExpr(LambdaExpr *LE) {
    if (!LE)
      return true;
    const CXXMethodDecl *CallOp = LE->getCallOperator();
    llvm::SaveAndRestore SaveSemantic(CurrentSemanticDC,
                                      CallOp ? CallOp : CurrentSemanticDC);
    llvm::SaveAndRestore SaveLexical(CurrentLexicalDC,
                                     CallOp ? CallOp : CurrentLexicalDC);
    llvm::SaveAndRestore<const DeclaratorDecl *> SaveOutOfLine(OutOfLineDecl,
                                                               nullptr);
    size_t SavedLocals = LocalDecls.size();
    for (const LambdaCapture &Cap : LE->captures()) {
      if (Cap.capturesVariable())
        if (ValueDecl *VD = Cap.getCapturedVar(); VD && VD->getIdentifier())
          LocalDecls.push_back({VD->getIdentifier(), Cap.getLocation()});
    }
    bool Res = RecursiveASTVisitor::TraverseLambdaExpr(LE);
    LocalDecls.truncate(SavedLocals);
    return Res;
  }

  bool VisitNamedDecl(NamedDecl *ND) {
    if (!ND || !ND->getIdentifier())
      return true;
    const DeclContext *DC = ND->getDeclContext();
    bool IsTemplateParam =
        llvm::isa<TemplateTypeParmDecl, NonTypeTemplateParmDecl,
                  TemplateTemplateParmDecl>(ND);
    if (IsTemplateParam ||
        (DC && !DC->isFileContext() && !DC->isRecord() &&
         !(llvm::isa<EnumDecl>(DC) && llvm::cast<EnumDecl>(DC)->isScoped())))
      LocalDecls.push_back({ND->getIdentifier(), ND->getLocation()});
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *D) {
    if (auto Ref = extractFromDeclRefExpr(D))
      maybeRewrite(*Ref);
    return !Err;
  }

  bool VisitOverloadExpr(OverloadExpr *OE) {
    if (OE->getName().getAsIdentifierInfo() && OE->getQualifierLoc()) {
      maybeRewrite(
          ExtractedQualifiedRef{OE->getQualifierLoc(),
                                SourceRange(OE->getQualifierLoc().getBeginLoc(),
                                            OE->getNameInfo().getLoc()),
                                SourceLocation()});
    }
    return !Err;
  }

  bool VisitTypeLoc(TypeLoc TL) {
    if (SkippedTypeLocStarts.contains(TL.getBeginLoc()))
      return true;
    if (auto Ref = extractFromTypeLoc(TL))
      maybeRewrite(*Ref);
    return !Err;
  }

private:
  struct LocalDeclEntry {
    const IdentifierInfo *Ident;
    SourceLocation Loc;
  };

  static bool isScopedStmt(const Stmt *S) {
    return llvm::isa<CompoundStmt, IfStmt, SwitchStmt, WhileStmt, ForStmt,
                     CXXForRangeStmt, CXXCatchStmt>(S);
  }

  bool isShadowedInEnclosingContexts(const DeclContext *StartDC) const {
    if (!StartDC || !InsertionPoint.EffectiveContext ||
        !InsertionPoint.EffectiveContext->Encloses(StartDC))
      return false;
    for (const DeclContext *DC = StartDC;
         DC && !DC->Equals(InsertionPoint.EffectiveContext);
         DC = DC->getParent()) {
      if (DC->isFunctionOrMethod())
        continue;
      if (const auto *NS = llvm::dyn_cast<NamespaceDecl>(DC)) {
        if (NS->getCanonicalDecl() == CanonicalTargetNS)
          continue;
      }
      for (const NamedDecl *Found : DC->lookup(TargetIdent)) {
        if (!isFromTargetNamespace(Found, CanonicalTargetNS))
          return true;
      }
      if (const auto *RD = llvm::dyn_cast<CXXRecordDecl>(DC)) {
        if (RD->hasDefinition()) {
          CXXBasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/false,
                             /*DetectVirtual=*/false);
          if (RD->lookupInBases(
                  [&](const CXXBaseSpecifier *, CXXBasePath &Path) {
                    return !Path.back().Class->lookup(TargetIdent).empty();
                  },
                  Paths))
            return true;
        }
      }
    }
    return false;
  }

  bool isShadowed(SourceLocation SpelledBeginLoc,
                  const DeclContext *LookupDC) const {
    for (const LocalDeclEntry &Entry : LocalDecls) {
      if (Entry.Ident != TargetIdent)
        continue;
      if (Entry.Loc.isInvalid() ||
          !SM.isBeforeInTranslationUnit(SpelledBeginLoc, Entry.Loc))
        return true;
    }
    return isShadowedInEnclosingContexts(LookupDC) ||
           isShadowedInEnclosingContexts(CurrentLexicalDC);
  }

  void maybeRewrite(const ExtractedQualifiedRef &Ref) {
    if (Err || !InsertionPoint.EffectiveContext)
      return;

    const auto *RefNS =
        getNamespaceFromQualifier(Ref.Qualifier.getNestedNameSpecifier());
    if (!RefNS || RefNS->getCanonicalDecl() != CanonicalTargetNS)
      return;

    if (SM.isMacroBodyExpansion(Ref.Qualifier.getBeginLoc()) ||
        !SM.isWrittenInSameFile(Ref.Qualifier.getBeginLoc(),
                                Ref.Qualifier.getEndLoc()))
      return;

    auto SpelledTokens =
        TB.spelledForExpanded(TB.expandedTokens(Ref.SpelledNameRange));
    if (!SpelledTokens || SpelledTokens->empty())
      return;

    SourceLocation SpelledBegin = SpelledTokens->front().location();
    SourceLocation SpelledNameLoc = SpelledTokens->back().location();
    if (SM.getFileID(SpelledBegin) != SM.getMainFileID() ||
        SM.getFileID(SpelledNameLoc) != SM.getMainFileID())
      return;

    // Leading specifiers/return types written before `Qual::` of an out-of-line
    // declaration are looked up in the lexical context rather than the
    // semantic context entered by `Qual::`.
    const DeclContext *LookupDC = CurrentSemanticDC;
    if (OutOfLineDecl &&
        SM.isBeforeInTranslationUnit(
            SpelledBegin, OutOfLineDecl->getQualifierLoc().getBeginLoc()))
      LookupDC = OutOfLineDecl->getLexicalDeclContext();
    if (!LookupDC || !InsertionPoint.EffectiveContext->Encloses(LookupDC))
      return;

    if (InsertionPoint.EffectiveLoc.isValid() &&
        SM.isBeforeInTranslationUnit(SpelledBegin, InsertionPoint.EffectiveLoc))
      return;

    if (SpelledTokens->back().text(SM) != TargetName)
      return;

    unsigned BeginOffset = SM.getFileOffset(SpelledBegin);
    unsigned EndOffset = SM.getFileOffset(SpelledNameLoc);
    if (BeginOffset >= EndOffset)
      return; // Synthesized typo-correction qualifier with nothing spelled.

    if (isShadowed(SpelledBegin, LookupDC))
      return;

    if (!SeenOffsets.insert(BeginOffset).second)
      return;

    Err = Repls.add(
        tooling::Replacement(SM, SpelledBegin, EndOffset - BeginOffset, ""));
  }

  const SourceManager &SM;
  const syntax::TokenBuffer &TB;
  const NamespaceDecl *CanonicalTargetNS;
  llvm::StringRef TargetName;
  const IdentifierInfo *TargetIdent;
  const InsertionPointData &InsertionPoint;
  tooling::Replacements &Repls;
  const DeclContext *CurrentSemanticDC;
  const DeclContext *CurrentLexicalDC;
  const DeclaratorDecl *OutOfLineDecl = nullptr;
  llvm::SmallVector<LocalDeclEntry, 8> LocalDecls;
  llvm::SmallVector<size_t, 8> ScopeStack;
  llvm::DenseSet<SourceLocation> SkippedTypeLocStarts;
  llvm::DenseSet<unsigned> SeenOffsets;
  llvm::Error Err = llvm::Error::success();
};

bool AddUsing::prepare(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();
  const auto &TB = Inputs.AST->getTokens();

  QualifierToRemove = NestedNameSpecifierLoc();
  QualifierToSpell.clear();
  SpelledQualifier = llvm::StringRef();
  SpelledName = llvm::StringRef();
  MustInsertAfterLoc = SourceLocation();

  // Do not suggest "using" in header files. That way madness lies.
  if (isHeaderFile(SM.getFileEntryRefForID(SM.getMainFileID())->getName(),
                   Inputs.AST->getLangOpts()))
    return false;

  auto *Node = Inputs.ASTSelection.commonAncestor();
  if (Node == nullptr)
    return false;

  // If we're looking at a type or NestedNameSpecifier, walk up the tree until
  // we find the "main" node we care about, which would be a qualified TypeLoc
  // or DeclRefExpr.
  for (; Node->Parent; Node = Node->Parent) {
    if (Node->ASTNode.get<NestedNameSpecifierLoc>()) {
      continue;
    }
    if (auto *T = Node->ASTNode.get<TypeLoc>()) {
      if (Node->Parent->ASTNode.get<NestedNameSpecifierLoc>())
        continue;
      if (isa<TagType, TemplateSpecializationType,
              DeducedTemplateSpecializationType, TypedefType, UsingType,
              UnresolvedUsingType>(T->getTypePtr()))
        break;
      // Find the outermost TypeLoc.
      if (Node->Parent->ASTNode.get<TypeLoc>())
        continue;
    }
    break;
  }
  if (Node == nullptr)
    return false;

  std::optional<ExtractedQualifiedRef> Extracted;
  if (auto *D = Node->ASTNode.get<DeclRefExpr>())
    Extracted = extractFromDeclRefExpr(D);
  else if (auto *T = Node->ASTNode.get<TypeLoc>())
    Extracted = extractFromTypeLoc(*T);

  if (!Extracted)
    return false;

  QualifierToRemove = Extracted->Qualifier;
  MustInsertAfterLoc = Extracted->MustInsertAfterLoc;

  if (!QualifierToRemove ||
      // FIXME: This only supports removing qualifiers that are made up of just
      // namespace names. If qualifier contains a type, we could take the
      // longest namespace prefix and remove that.
      QualifierToRemove.getNestedNameSpecifier().getKind() !=
          NestedNameSpecifier::Kind::Namespace ||
      // Respect user config.
      isNamespaceForbidden(Inputs, QualifierToRemove.getNestedNameSpecifier()))
    return false;
  // Macros are difficult. We only want to offer code action when what's spelled
  // under the cursor is a namespace qualifier. If it's a macro that expands to
  // a qualifier, user would not know what code action will actually change.
  // On the other hand, if the qualifier is part of the macro argument, we
  // should still support that.
  if (SM.isMacroBodyExpansion(QualifierToRemove.getBeginLoc()) ||
      !SM.isWrittenInSameFile(QualifierToRemove.getBeginLoc(),
                              QualifierToRemove.getEndLoc())) {
    return false;
  }

  auto SpelledTokens =
      TB.spelledForExpanded(TB.expandedTokens(Extracted->SpelledNameRange));
  if (!SpelledTokens || SpelledTokens->empty())
    return false;
  auto SpelledRange =
      syntax::Token::range(SM, SpelledTokens->front(), SpelledTokens->back());
  // We only drop qualifiers that're namespaces, so this is safe.
  std::tie(SpelledQualifier, SpelledName) =
      splitQualifiedName(SpelledRange.text(SM));
  QualifierToSpell = getNNSLAsString(
      QualifierToRemove, Inputs.AST->getASTContext().getPrintingPolicy());
  if (!llvm::StringRef(QualifierToSpell).ends_with(SpelledQualifier) ||
      SpelledName.empty())
    return false; // What's spelled doesn't match the qualifier.
  return true;
}

Expected<Tweak::Effect> AddUsing::apply(const Selection &Inputs) {
  auto &ASTCtx = Inputs.AST->getASTContext();
  auto &SM = ASTCtx.getSourceManager();

  const auto *TargetNS =
      getNamespaceFromQualifier(QualifierToRemove.getNestedNameSpecifier());
  if (!TargetNS)
    return error("Qualifier is not a namespace");

  auto InsertionPoint = findInsertionPoint(Inputs, QualifierToRemove,
                                           SpelledName, MustInsertAfterLoc);
  if (!InsertionPoint) {
    return InsertionPoint.takeError();
  }

  tooling::Replacements R;
  if (InsertionPoint->Loc.isValid()) {
    // Add the using statement at appropriate location.
    std::string UsingText;
    llvm::raw_string_ostream UsingTextStream(UsingText);
    UsingTextStream << "using ";
    if (InsertionPoint->AlwaysFullyQualify &&
        !QualifierToRemove.getNestedNameSpecifier().isFullyQualified())
      UsingTextStream << "::";
    UsingTextStream << QualifierToSpell << SpelledName << ";"
                    << InsertionPoint->Suffix;

    assert(SM.getFileID(InsertionPoint->Loc) == SM.getMainFileID());
    if (auto Err = R.add(tooling::Replacement(SM, InsertionPoint->Loc, 0,
                                              UsingTextStream.str()))) {
      return std::move(Err);
    }
  }

  QualifiedRefCollector Collector(ASTCtx, Inputs.AST->getTokens(),
                                  TargetNS->getCanonicalDecl(), SpelledName,
                                  *InsertionPoint, R);
  if (InsertionPoint->LocalScope) {
    Collector.TraverseStmt(const_cast<Stmt *>(InsertionPoint->LocalScope));
  } else {
    for (Decl *D : Inputs.AST->getLocalTopLevelDecls())
      if (!Collector.TraverseDecl(D))
        break;
  }
  if (auto Err = Collector.takeError())
    return std::move(Err);

  return Effect::mainFileEdit(SM, std::move(R));
}

} // namespace
} // namespace clangd
} // namespace clang
