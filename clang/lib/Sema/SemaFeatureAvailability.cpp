//===--- SemaFeatureAvailability.cpp - Availability attribute handling ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file processes the feature availability attribute.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallSet.h"
#include <utility>

using namespace clang;
using namespace sema;

static bool isFeatureUseGuarded(const DomainAvailabilityAttr *AA,
                                const Decl *ContextDecl, ASTContext &Ctx) {
  for (auto *Attr : ContextDecl->specific_attrs<DomainAvailabilityAttr>())
    if (AA->getDomain() == Attr->getDomain())
      return AA->getUnavailable() == Attr->getUnavailable();
  return false;
}

static void diagnoseDeclFeatureAvailability(const NamedDecl *D,
                                            SourceLocation Loc,
                                            Decl *ContextDecl, Sema &S) {
  for (auto *Attr : D->specific_attrs<DomainAvailabilityAttr>()) {
    std::string FeatureUse = Attr->getDomain().str();
    // Skip checking if the feature is always enabled.
    if (!Attr->getUnavailable() &&
        S.Context.getFeatureAvailInfo(FeatureUse).Kind ==
            FeatureAvailKind::AlwaysAvailable)
      continue;

    if (!isFeatureUseGuarded(Attr, ContextDecl, S.Context))
      S.Diag(Loc, diag::err_unguarded_feature)
          << D << FeatureUse << Attr->getUnavailable();
  }
}

class DiagnoseUnguardedFeatureAvailability
    : public RecursiveASTVisitor<DiagnoseUnguardedFeatureAvailability> {

  typedef RecursiveASTVisitor<DiagnoseUnguardedFeatureAvailability> Base;

  Sema &SemaRef;
  const Decl *D;

  struct FeatureAvailInfo {
    StringRef Domain;
    bool Unavailable;
  };

  SmallVector<const Stmt *, 16> StmtStack;
  SmallVector<FeatureAvailInfo, 4> FeatureStack;

  bool isFeatureUseGuarded(const DomainAvailabilityAttr *Attr) const;

  bool isConditionallyGuardedByFeature() const;

public:
  DiagnoseUnguardedFeatureAvailability(Sema &SemaRef, const Decl *D,
                                       Decl *Ctx = nullptr)
      : SemaRef(SemaRef), D(D) {}

  void diagnoseDeclFeatureAvailability(const NamedDecl *D, SourceRange Range);

  bool TraverseStmt(Stmt *S) {
    if (!S)
      return true;
    StmtStack.push_back(S);
    bool Result = Base::TraverseStmt(S);
    StmtStack.pop_back();
    return Result;
  }

  bool TraverseIfStmt(IfStmt *If);

  // Ignore unguarded uses of enumerators inside case label expressions.
  bool TraverseCaseStmt(CaseStmt *Case) {
    return TraverseStmt(Case->getSubStmt());
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    diagnoseDeclFeatureAvailability(DRE->getDecl(), DRE->getSourceRange());
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    diagnoseDeclFeatureAvailability(ME->getMemberDecl(), ME->getSourceRange());
    return true;
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *OME) {
    if (auto *MD = OME->getMethodDecl())
      diagnoseDeclFeatureAvailability(MD, OME->getSourceRange());
    return true;
  }

  bool VisitTypeLoc(TypeLoc Ty);

  void IssueDiagnostics() {
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      TraverseStmt(FD->getBody());
    else if (auto *OMD = dyn_cast<ObjCMethodDecl>(D))
      TraverseStmt(OMD->getBody());
  }
};

static std::pair<StringRef, bool> extractFeatureExpr(const Expr *IfCond) {
  const auto *E = IfCond;
  bool IsNegated = false;
  while (true) {
    E = E->IgnoreParens();
    if (const auto *AE = dyn_cast<ObjCAvailabilityCheckExpr>(E)) {
      if (!AE->hasDomainName())
        return {};
      return {AE->getDomainName(), IsNegated};
    }

    const auto *UO = dyn_cast<UnaryOperator>(E);
    if (!UO || UO->getOpcode() != UO_LNot) {
      return {};
    }
    E = UO->getSubExpr();
    IsNegated = !IsNegated;
  }
}

bool DiagnoseUnguardedFeatureAvailability::isConditionallyGuardedByFeature()
    const {
  return FeatureStack.size();
}

bool DiagnoseUnguardedFeatureAvailability::TraverseIfStmt(IfStmt *If) {
  std::pair<StringRef, bool> IfCond;
  if (auto *Cond = If->getCond())
    IfCond = extractFeatureExpr(Cond);
  if (IfCond.first.empty()) {
    // This isn't an availability checking 'if', we can just continue.
    return Base::TraverseIfStmt(If);
  }

  StringRef FeatureStr = IfCond.first;
  auto *Guarded = If->getThen();
  auto *Unguarded = If->getElse();
  if (IfCond.second) {
    std::swap(Guarded, Unguarded);
  }

  FeatureStack.push_back({FeatureStr, false});
  bool ShouldContinue = TraverseStmt(Guarded);
  FeatureStack.pop_back();

  if (!ShouldContinue)
    return false;

  FeatureStack.push_back({FeatureStr, true});
  ShouldContinue = TraverseStmt(Unguarded);
  FeatureStack.pop_back();
  return ShouldContinue;
}

bool DiagnoseUnguardedFeatureAvailability::isFeatureUseGuarded(
    const DomainAvailabilityAttr *Attr) const {
  auto Domain = Attr->getDomain();
  for (auto &Info : FeatureStack)
    if (Info.Domain == Domain && Info.Unavailable == Attr->getUnavailable())
      return true;
  return ::isFeatureUseGuarded(Attr, D, SemaRef.Context);
}

/// Returns true if the given statement can be a body-like child of \p Parent.
bool isBodyLikeChildStmt(const Stmt *S, const Stmt *Parent) {
  switch (Parent->getStmtClass()) {
  case Stmt::IfStmtClass:
    return cast<IfStmt>(Parent)->getThen() == S ||
           cast<IfStmt>(Parent)->getElse() == S;
  case Stmt::WhileStmtClass:
    return cast<WhileStmt>(Parent)->getBody() == S;
  case Stmt::DoStmtClass:
    return cast<DoStmt>(Parent)->getBody() == S;
  case Stmt::ForStmtClass:
    return cast<ForStmt>(Parent)->getBody() == S;
  case Stmt::CXXForRangeStmtClass:
    return cast<CXXForRangeStmt>(Parent)->getBody() == S;
  case Stmt::ObjCForCollectionStmtClass:
    return cast<ObjCForCollectionStmt>(Parent)->getBody() == S;
  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
    return cast<SwitchCase>(Parent)->getSubStmt() == S;
  default:
    return false;
  }
}

/// Traverses the AST and finds the last statement that used a declaration in
/// the given list.
class LastDeclUSEFinder : public RecursiveASTVisitor<LastDeclUSEFinder> {
  llvm::SmallSet<const Decl *, 4> Decls;

public:
  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (Decls.contains(DRE->getDecl()))
      return false;
    return true;
  }

  static const Stmt *findLastStmtThatUsesDecl(const DeclStmt *UseStmt,
                                              const CompoundStmt *Scope) {
    LastDeclUSEFinder Visitor;
    Visitor.Decls.insert(UseStmt->decl_begin(), UseStmt->decl_end());
    const Stmt *LastUse = UseStmt;
    for (const Stmt *S : Scope->body()) {
      if (!Visitor.TraverseStmt(const_cast<Stmt *>(S)))
        LastUse = S;
      if (auto *DS = dyn_cast<DeclStmt>(S))
        Visitor.Decls.insert(DS->decl_begin(), DS->decl_end());
    }
    return LastUse;
  }
};

void DiagnoseUnguardedFeatureAvailability::diagnoseDeclFeatureAvailability(
    const NamedDecl *D, SourceRange Range) {
  SourceLocation Loc = Range.getBegin();
  SmallVector<const DomainAvailabilityAttr *, 1> Attrs;
  for (auto *Attr : D->specific_attrs<DomainAvailabilityAttr>()) {
    std::string FeatureUse = Attr->getDomain().str();
    // Skip checking if the feature is always enabled.
    if (!Attr->getUnavailable() &&
        SemaRef.Context.getFeatureAvailInfo(FeatureUse).Kind ==
            FeatureAvailKind::AlwaysAvailable)
      continue;

    if (!isFeatureUseGuarded(Attr)) {
      SemaRef.Diag(Loc, diag::err_unguarded_feature)
          << D << FeatureUse << Attr->getUnavailable();
      Attrs.push_back(Attr);
    }
  }

  if (Attrs.empty())
    return;

  auto FixitDiag =
      SemaRef.Diag(Range.getBegin(),
                   diag::note_silence_unguarded_availability_domain)
      << Range << D
      << (SemaRef.getLangOpts().ObjC ? /*@available*/ 0
                                     : /*__builtin_available*/ 1);

  // Find the statement which should be enclosed in the if @available check.
  if (StmtStack.empty())
    return;

  const Stmt *StmtOfUse = StmtStack.back();
  const CompoundStmt *Scope = nullptr;
  for (const Stmt *S : llvm::reverse(StmtStack)) {
    if (const auto *CS = dyn_cast<CompoundStmt>(S)) {
      Scope = CS;
      break;
    }
    if (isBodyLikeChildStmt(StmtOfUse, S)) {
      // The declaration won't be seen outside of the statement, so we don't
      // have to wrap the uses of any declared variables in if (@available).
      // Therefore we can avoid setting Scope here.
      break;
    }

    StmtOfUse = S;
  }

  const Stmt *LastStmtOfUse = nullptr;
  if (auto *DS = dyn_cast<DeclStmt>(StmtOfUse); DS && Scope)
    LastStmtOfUse = LastDeclUSEFinder::findLastStmtThatUsesDecl(DS, Scope);

  const SourceManager &SM = SemaRef.getSourceManager();
  SourceLocation IfInsertionLoc = SM.getExpansionLoc(StmtOfUse->getBeginLoc());
  SourceLocation StmtEndLoc =
      SM.getExpansionRange(
            (LastStmtOfUse ? LastStmtOfUse : StmtOfUse)->getEndLoc())
          .getEnd();

  if (SM.getFileID(IfInsertionLoc) != SM.getFileID(StmtEndLoc))
    return;

  std::string Prefix;
  llvm::raw_string_ostream FixItOS(Prefix);
  StringRef AvailStr =
      SemaRef.getLangOpts().ObjC ? "@available" : "__builtin_available";
  for (auto *A : Attrs) {
    FixItOS << "if (" << AvailStr << "(domain:" << A->getDomain() << ")) {";
    if (A->getUnavailable())
      FixItOS << "} else {";
  }

  FixitDiag << FixItHint::CreateInsertion(IfInsertionLoc, FixItOS.str());
  FixItOS.str().clear();

  for (auto *A : llvm::reverse(Attrs)) {
    FixItOS << "}";
    if (!A->getUnavailable())
      FixItOS << " else {}";
  }

  SourceLocation ElseInsertionLoc = Lexer::findLocationAfterToken(
      StmtEndLoc, tok::semi, SM, SemaRef.getLangOpts(),
      /*SkipTrailingWhitespaceAndNewLine=*/false);
  if (ElseInsertionLoc.isInvalid())
    ElseInsertionLoc =
        Lexer::getLocForEndOfToken(StmtEndLoc, 0, SM, SemaRef.getLangOpts());
  FixitDiag << FixItHint::CreateInsertion(ElseInsertionLoc, FixItOS.str());
}

bool DiagnoseUnguardedFeatureAvailability::VisitTypeLoc(TypeLoc Ty) {
  const Type *TyPtr = Ty.getTypePtr();
  SourceLocation Loc = Ty.getBeginLoc();

  if (Loc.isInvalid())
    return true;

  if (const auto *TT = dyn_cast<TagType>(TyPtr)) {
    TagDecl *TD = TT->getDecl();
    diagnoseDeclFeatureAvailability(TD, Ty.getSourceRange());
  } else if (const auto *TD = dyn_cast<TypedefType>(TyPtr)) {
    TypedefNameDecl *D = TD->getDecl();
    diagnoseDeclFeatureAvailability(D, Ty.getSourceRange());
  } else if (const auto *ObjCO = dyn_cast<ObjCObjectType>(TyPtr)) {
    if (NamedDecl *D = ObjCO->getInterface())
      diagnoseDeclFeatureAvailability(D, Ty.getSourceRange());
  }

  return true;
}

void Sema::handleDelayedFeatureAvailabilityCheck(DelayedDiagnostic &DD,
                                                 Decl *Ctx) {
  assert(DD.Kind == DelayedDiagnostic::FeatureAvailability &&
         "Expected a feature availability diagnostic here");

  DD.Triggered = true;
  diagnoseDeclFeatureAvailability(DD.getFeatureAvailabilityDecl(), DD.Loc, Ctx,
                                  *this);
}

void Sema::DiagnoseUnguardedFeatureAvailabilityViolations(Decl *D) {
  assert((D->getAsFunction() || isa<ObjCMethodDecl>(D)) &&
         "function or ObjC method decl expected");
  DiagnoseUnguardedFeatureAvailability(*this, D).IssueDiagnostics();
}

void Sema::DiagnoseFeatureAvailabilityOfDecl(NamedDecl *D,
                                             ArrayRef<SourceLocation> Locs) {
  if (!Context.hasFeatureAvailabilityAttr(D))
    return;

  if (FunctionScopeInfo *Context = getCurFunctionAvailabilityContext()) {
    Context->HasPotentialFeatureAvailabilityViolations = true;
    return;
  }

  if (DelayedDiagnostics.shouldDelayDiagnostics()) {
    DelayedDiagnostics.add(DelayedDiagnostic::makeFeatureAvailability(D, Locs));
    return;
  }

  Decl *Ctx = cast<Decl>(getCurLexicalContext());
  diagnoseDeclFeatureAvailability(D, Locs.front(), Ctx, *this);
}

static bool isSimpleFeatureAvailabiltyMacro(MacroInfo *Info) {
  // Must match:
  // __attribute__((availability(domain : id, id/numeric_constant)))
  if (Info->getNumTokens() != 13)
    return false;

  if (!Info->getReplacementToken(0).is(tok::kw___attribute) ||
      !Info->getReplacementToken(1).is(tok::l_paren) ||
      !Info->getReplacementToken(2).is(tok::l_paren))
    return false;

  if (const Token &Tk = Info->getReplacementToken(3);
      !Tk.is(tok::identifier) ||
      Tk.getIdentifierInfo()->getName() != "availability")
    return false;

  if (!Info->getReplacementToken(4).is(tok::l_paren))
    return false;

  if (const Token &Tk = Info->getReplacementToken(5);
      !Tk.is(tok::identifier) || Tk.getIdentifierInfo()->getName() != "domain")
    return false;

  if (!Info->getReplacementToken(6).is(tok::colon))
    return false;

  if (const Token &Tk = Info->getReplacementToken(7); !Tk.is(tok::identifier))
    return false;

  if (!Info->getReplacementToken(8).is(tok::comma))
    return false;

  if (const Token &Tk = Info->getReplacementToken(9);
      !Tk.is(tok::identifier) && !Tk.is(tok::numeric_constant))
    return false;

  if (!Info->getReplacementToken(10).is(tok::r_paren) ||
      !Info->getReplacementToken(11).is(tok::r_paren) ||
      !Info->getReplacementToken(12).is(tok::r_paren))
    return false;

  return true;
}

void Sema::diagnoseDeprecatedAvailabilityDomain(StringRef DomainName,
                                                SourceLocation AvailLoc,
                                                SourceLocation DomainLoc,
                                                bool IsUnavailable,
                                                const ParsedAttr *PA) {
  auto CreateFixIt = [&]() {
    if (PA->getRange().getBegin().isMacroID()) {
      auto *MacroII = PA->getMacroIdentifier();

      // Macro identifier isn't always set.
      if (!MacroII)
        return FixItHint{};

      MacroDefinition MD = PP.getMacroDefinition(MacroII);
      MacroInfo *Info = MD.getMacroInfo();

      bool IsSimple;
      auto It = SimpleFeatureAvailabiltyMacros.find(Info);

      if (It == SimpleFeatureAvailabiltyMacros.end()) {
        IsSimple = isSimpleFeatureAvailabiltyMacro(Info);
        SimpleFeatureAvailabiltyMacros[Info] = IsSimple;
      } else {
        IsSimple = It->second;
      }

      if (!IsSimple)
        return FixItHint{};

      FileID FID = SourceMgr.getFileID(AvailLoc);
      const SrcMgr::ExpansionInfo *EI =
          &SourceMgr.getSLocEntry(FID).getExpansion();
      if (IsUnavailable)
        return FixItHint::CreateReplacement(EI->getExpansionLocRange(),
                                            "__attribute__((unavailable))");
      return FixItHint::CreateRemoval(EI->getExpansionLocRange());
    }

    if (PA->getSyntax() != AttributeCommonInfo::Syntax::AS_GNU)
      return FixItHint{};

    SourceRange AttrRange = PA->getRange();

    // Replace the availability attribute with "unavailable".
    if (IsUnavailable)
      return FixItHint::CreateReplacement(AttrRange, "unavailable");

    // Remove the availability attribute.

    // If there is a leading comma, there's another operand that precedes the
    // availability attribute. In that case, remove the availability attribute
    // and the comma.
    Token PrevTok = *Lexer::findPreviousToken(AttrRange.getBegin(), SourceMgr,
                                              getLangOpts(), false);
    if (PrevTok.is(tok::comma))
      return FixItHint::CreateRemoval(
          SourceRange(PrevTok.getLocation(), AttrRange.getEnd()));

    // If there is a trailing comma, there's another operand that follows the
    // availability attribute. In that case, remove the availability attribute
    // and the comma.
    Token NextTok = *Lexer::findNextToken(AttrRange.getEnd(), SourceMgr,
                                          getLangOpts(), false);
    if (NextTok.is(tok::comma))
      return FixItHint::CreateRemoval(
          SourceRange(AttrRange.getBegin(), NextTok.getLocation()));

    // If no leading or trailing commas are found, the availability attribute is
    // the only operand. Remove the entire attribute construct.

    // Look for '__attribute'.
    for (int i = 0; i < 2; ++i)
      PrevTok = *Lexer::findPreviousToken(PrevTok.getLocation(), SourceMgr,
                                          getLangOpts(), false);
    if (!PrevTok.is(tok::raw_identifier) ||
        PrevTok.getRawIdentifier() != "__attribute__")
      return FixItHint{};

    // Look for the closing ')'.
    NextTok = *Lexer::findNextToken(NextTok.getLocation(), SourceMgr,
                                    getLangOpts(), false);
    if (!NextTok.is(tok::r_paren))
      return FixItHint{};

    return FixItHint::CreateRemoval(
        SourceRange(PrevTok.getLocation(), NextTok.getLocation()));
  };

  ASTContext::AvailabilityDomainInfo Info =
      Context.getFeatureAvailInfo(DomainName);

  if (Info.IsDeprecated) {
    Diag(DomainLoc, diag::warn_deprecated_availability_domain) << DomainName;
    if (Info.Kind == FeatureAvailKind::AlwaysAvailable) {
      if (PA) {
        auto FixitDiag =
            Diag(AvailLoc, diag::warn_permanently_available_domain_decl)
            << DomainName << IsUnavailable;

        FixItHint Hint = CreateFixIt();
        if (!Hint.isNull())
          FixitDiag << Hint;
      } else
        Diag(AvailLoc, diag::warn_permanently_available_domain_expr)
            << DomainName;
    }
  }
}
