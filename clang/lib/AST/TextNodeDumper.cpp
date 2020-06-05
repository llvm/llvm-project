//===--- TextNodeDumper.cpp - Printing of AST nodes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements AST dumping of components of individual AST nodes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/TextNodeDumper.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/LocInfoType.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"

using namespace clang;

static void dumpPreviousDeclImpl(raw_ostream &OS, ...) {}

template <typename T>
static void dumpPreviousDeclImpl(raw_ostream &OS, const Mergeable<T> *D) {
  const T *First = D->getFirstDecl();
  if (First != D)
    OS << " first " << First;
}

template <typename T>
static void dumpPreviousDeclImpl(raw_ostream &OS, const Redeclarable<T> *D) {
  const T *Prev = D->getPreviousDecl();
  if (Prev)
    OS << " prev " << Prev;
}

/// Dump the previous declaration in the redeclaration chain for a declaration,
/// if any.
static void dumpPreviousDecl(raw_ostream &OS, const Decl *D) {
  switch (D->getKind()) {
#define DECL(DERIVED, BASE)                                                    \
  case Decl::DERIVED:                                                          \
    return dumpPreviousDeclImpl(OS, cast<DERIVED##Decl>(D));
#define ABSTRACT_DECL(DECL)
#include "clang/AST/DeclNodes.inc"
  }
  llvm_unreachable("Decl that isn't part of DeclNodes.inc!");
}

TextNodeDumper::TextNodeDumper(raw_ostream &OS, bool ShowColors,
                               const SourceManager *SM,
                               const PrintingPolicy &PrintPolicy,
                               const comments::CommandTraits *Traits)
    : TextTreeStructure(OS, ShowColors), OS(OS), ShowColors(ShowColors), SM(SM),
      PrintPolicy(PrintPolicy), Traits(Traits) {}

void TextNodeDumper::Visit(const comments::Comment *C,
                           const comments::FullComment *FC) {
  if (!C) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  {
    ColorScope Color(OS, ShowColors, CommentColor);
    OS << C->getCommentKindName();
  }
  dumpPointer(C);
  dumpSourceRange(C->getSourceRange());

  ConstCommentVisitor<TextNodeDumper, void,
                      const comments::FullComment *>::visit(C, FC);
}

void TextNodeDumper::Visit(const Attr *A) {
  {
    ColorScope Color(OS, ShowColors, AttrColor);

    switch (A->getKind()) {
#define ATTR(X)                                                                \
  case attr::X:                                                                \
    OS << #X;                                                                  \
    break;
#include "clang/Basic/AttrList.inc"
    }
    OS << "Attr";
  }
  dumpPointer(A);
  dumpSourceRange(A->getRange());
  if (A->isInherited())
    OS << " Inherited";
  if (A->isImplicit())
    OS << " Implicit";

  ConstAttrVisitor<TextNodeDumper>::Visit(A);
}

void TextNodeDumper::Visit(const TemplateArgument &TA, SourceRange R,
                           const Decl *From, StringRef Label) {
  OS << "TemplateArgument";
  if (R.isValid())
    dumpSourceRange(R);

  if (From)
    dumpDeclRef(From, Label);

  ConstTemplateArgumentVisitor<TextNodeDumper>::Visit(TA);
}

void TextNodeDumper::Visit(const Stmt *Node) {
  if (!Node) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>>";
    return;
  }
  {
    ColorScope Color(OS, ShowColors, StmtColor);
    OS << Node->getStmtClassName();
  }
  dumpPointer(Node);
  dumpSourceRange(Node->getSourceRange());

  if (const auto *E = dyn_cast<Expr>(Node)) {
    dumpType(E->getType());

    if (E->containsErrors()) {
      ColorScope Color(OS, ShowColors, ErrorsColor);
      OS << " contains-errors";
    }

    {
      ColorScope Color(OS, ShowColors, ValueKindColor);
      switch (E->getValueKind()) {
      case VK_RValue:
        break;
      case VK_LValue:
        OS << " lvalue";
        break;
      case VK_XValue:
        OS << " xvalue";
        break;
      }
    }

    {
      ColorScope Color(OS, ShowColors, ObjectKindColor);
      switch (E->getObjectKind()) {
      case OK_Ordinary:
        break;
      case OK_BitField:
        OS << " bitfield";
        break;
      case OK_ObjCProperty:
        OS << " objcproperty";
        break;
      case OK_ObjCSubscript:
        OS << " objcsubscript";
        break;
      case OK_VectorComponent:
        OS << " vectorcomponent";
        break;
      case OK_MatrixComponent:
        OS << " matrixcomponent";
        break;
      }
    }
  }

  ConstStmtVisitor<TextNodeDumper>::Visit(Node);
}

void TextNodeDumper::Visit(const Type *T) {
  if (!T) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>>";
    return;
  }
  if (isa<LocInfoType>(T)) {
    {
      ColorScope Color(OS, ShowColors, TypeColor);
      OS << "LocInfo Type";
    }
    dumpPointer(T);
    return;
  }

  {
    ColorScope Color(OS, ShowColors, TypeColor);
    OS << T->getTypeClassName() << "Type";
  }
  dumpPointer(T);
  OS << " ";
  dumpBareType(QualType(T, 0), false);

  QualType SingleStepDesugar =
      T->getLocallyUnqualifiedSingleStepDesugaredType();
  if (SingleStepDesugar != QualType(T, 0))
    OS << " sugar";

  if (T->isDependentType())
    OS << " dependent";
  else if (T->isInstantiationDependentType())
    OS << " instantiation_dependent";

  if (T->isVariablyModifiedType())
    OS << " variably_modified";
  if (T->containsUnexpandedParameterPack())
    OS << " contains_unexpanded_pack";
  if (T->isFromAST())
    OS << " imported";

  TypeVisitor<TextNodeDumper>::Visit(T);
}

void TextNodeDumper::Visit(QualType T) {
  OS << "QualType";
  dumpPointer(T.getAsOpaquePtr());
  OS << " ";
  dumpBareType(T, false);
  OS << " " << T.split().Quals.getAsString();
}

void TextNodeDumper::Visit(const Decl *D) {
  if (!D) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  {
    ColorScope Color(OS, ShowColors, DeclKindNameColor);
    OS << D->getDeclKindName() << "Decl";
  }
  dumpPointer(D);
  if (D->getLexicalDeclContext() != D->getDeclContext())
    OS << " parent " << cast<Decl>(D->getDeclContext());
  dumpPreviousDecl(OS, D);
  dumpSourceRange(D->getSourceRange());
  OS << ' ';
  dumpLocation(D->getLocation());
  if (D->isFromASTFile())
    OS << " imported";
  if (Module *M = D->getOwningModule())
    OS << " in " << M->getFullModuleName();
  if (auto *ND = dyn_cast<NamedDecl>(D))
    for (Module *M : D->getASTContext().getModulesWithMergedDefinition(
             const_cast<NamedDecl *>(ND)))
      AddChild([=] { OS << "also in " << M->getFullModuleName(); });
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
    if (ND->isHidden())
      OS << " hidden";
  if (D->isImplicit())
    OS << " implicit";

  if (D->isUsed())
    OS << " used";
  else if (D->isThisDeclarationReferenced())
    OS << " referenced";

  if (D->isInvalidDecl())
    OS << " invalid";
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isConstexprSpecified())
      OS << " constexpr";
    if (FD->isConsteval())
      OS << " consteval";
  }

  if (!isa<FunctionDecl>(*D)) {
    const auto *MD = dyn_cast<ObjCMethodDecl>(D);
    if (!MD || !MD->isThisDeclarationADefinition()) {
      const auto *DC = dyn_cast<DeclContext>(D);
      if (DC && DC->hasExternalLexicalStorage()) {
        ColorScope Color(OS, ShowColors, UndeserializedColor);
        OS << " <undeserialized declarations>";
      }
    }
  }

  ConstDeclVisitor<TextNodeDumper>::Visit(D);
}

void TextNodeDumper::Visit(const CXXCtorInitializer *Init) {
  OS << "CXXCtorInitializer";
  if (Init->isAnyMemberInitializer()) {
    OS << ' ';
    dumpBareDeclRef(Init->getAnyMember());
  } else if (Init->isBaseInitializer()) {
    dumpType(QualType(Init->getBaseClass(), 0));
  } else if (Init->isDelegatingInitializer()) {
    dumpType(Init->getTypeSourceInfo()->getType());
  } else {
    llvm_unreachable("Unknown initializer type");
  }
}

void TextNodeDumper::Visit(const BlockDecl::Capture &C) {
  OS << "capture";
  if (C.isByRef())
    OS << " byref";
  if (C.isNested())
    OS << " nested";
  if (C.getVariable()) {
    OS << ' ';
    dumpBareDeclRef(C.getVariable());
  }
}

void TextNodeDumper::Visit(const OMPClause *C) {
  if (!C) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>> OMPClause";
    return;
  }
  {
    ColorScope Color(OS, ShowColors, AttrColor);
    StringRef ClauseName(llvm::omp::getOpenMPClauseName(C->getClauseKind()));
    OS << "OMP" << ClauseName.substr(/*Start=*/0, /*N=*/1).upper()
       << ClauseName.drop_front() << "Clause";
  }
  dumpPointer(C);
  dumpSourceRange(SourceRange(C->getBeginLoc(), C->getEndLoc()));
  if (C->isImplicit())
    OS << " <implicit>";
}

void TextNodeDumper::Visit(const GenericSelectionExpr::ConstAssociation &A) {
  const TypeSourceInfo *TSI = A.getTypeSourceInfo();
  if (TSI) {
    OS << "case ";
    dumpType(TSI->getType());
  } else {
    OS << "default";
  }

  if (A.isSelected())
    OS << " selected";
}

void TextNodeDumper::dumpPointer(const void *Ptr) {
  ColorScope Color(OS, ShowColors, AddressColor);
  OS << ' ' << Ptr;
}

void TextNodeDumper::dumpLocation(SourceLocation Loc) {
  if (!SM)
    return;

  ColorScope Color(OS, ShowColors, LocationColor);
  SourceLocation SpellingLoc = SM->getSpellingLoc(Loc);

  // The general format we print out is filename:line:col, but we drop pieces
  // that haven't changed since the last loc printed.
  PresumedLoc PLoc = SM->getPresumedLoc(SpellingLoc);

  if (PLoc.isInvalid()) {
    OS << "<invalid sloc>";
    return;
  }

  if (strcmp(PLoc.getFilename(), LastLocFilename) != 0) {
    OS << PLoc.getFilename() << ':' << PLoc.getLine() << ':'
       << PLoc.getColumn();
    LastLocFilename = PLoc.getFilename();
    LastLocLine = PLoc.getLine();
  } else if (PLoc.getLine() != LastLocLine) {
    OS << "line" << ':' << PLoc.getLine() << ':' << PLoc.getColumn();
    LastLocLine = PLoc.getLine();
  } else {
    OS << "col" << ':' << PLoc.getColumn();
  }
}

void TextNodeDumper::dumpSourceRange(SourceRange R) {
  // Can't translate locations if a SourceManager isn't available.
  if (!SM)
    return;

  OS << " <";
  dumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    OS << ", ";
    dumpLocation(R.getEnd());
  }
  OS << ">";

  // <t2.c:123:421[blah], t2.c:412:321>
}

void TextNodeDumper::dumpBareType(QualType T, bool Desugar) {
  ColorScope Color(OS, ShowColors, TypeColor);

  SplitQualType T_split = T.split();
  OS << "'" << QualType::getAsString(T_split, PrintPolicy) << "'";

  if (Desugar && !T.isNull()) {
    // If the type is sugared, also dump a (shallow) desugared type.
    SplitQualType D_split = T.getSplitDesugaredType();
    if (T_split != D_split)
      OS << ":'" << QualType::getAsString(D_split, PrintPolicy) << "'";
  }
}

void TextNodeDumper::dumpType(QualType T) {
  OS << ' ';
  dumpBareType(T);
}

void TextNodeDumper::dumpBareDeclRef(const Decl *D) {
  if (!D) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  {
    ColorScope Color(OS, ShowColors, DeclKindNameColor);
    OS << D->getDeclKindName();
  }
  dumpPointer(D);

  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    ColorScope Color(OS, ShowColors, DeclNameColor);
    OS << " '" << ND->getDeclName() << '\'';
  }

  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D))
    dumpType(VD->getType());
}

void TextNodeDumper::dumpName(const NamedDecl *ND) {
  if (ND->getDeclName()) {
    ColorScope Color(OS, ShowColors, DeclNameColor);
    OS << ' ' << ND->getNameAsString();
  }
}

void TextNodeDumper::dumpAccessSpecifier(AccessSpecifier AS) {
  const auto AccessSpelling = getAccessSpelling(AS);
  if (AccessSpelling.empty())
    return;
  OS << AccessSpelling;
}

void TextNodeDumper::dumpCleanupObject(
    const ExprWithCleanups::CleanupObject &C) {
  if (auto *BD = C.dyn_cast<BlockDecl *>())
    dumpDeclRef(BD, "cleanup");
  else if (auto *CLE = C.dyn_cast<CompoundLiteralExpr *>())
    AddChild([=] {
      OS << "cleanup ";
      {
        ColorScope Color(OS, ShowColors, StmtColor);
        OS << CLE->getStmtClassName();
      }
      dumpPointer(CLE);
    });
  else
    llvm_unreachable("unexpected cleanup type");
}

void TextNodeDumper::dumpDeclRef(const Decl *D, StringRef Label) {
  if (!D)
    return;

  AddChild([=] {
    if (!Label.empty())
      OS << Label << ' ';
    dumpBareDeclRef(D);
  });
}

const char *TextNodeDumper::getCommandName(unsigned CommandID) {
  if (Traits)
    return Traits->getCommandInfo(CommandID)->Name;
  const comments::CommandInfo *Info =
      comments::CommandTraits::getBuiltinCommandInfo(CommandID);
  if (Info)
    return Info->Name;
  return "<not a builtin command>";
}

void TextNodeDumper::visitTextComment(const comments::TextComment *C,
                                      const comments::FullComment *) {
  OS << " Text=\"" << C->getText() << "\"";
}

void TextNodeDumper::visitInlineCommandComment(
    const comments::InlineCommandComment *C, const comments::FullComment *) {
  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  switch (C->getRenderKind()) {
  case comments::InlineCommandComment::RenderNormal:
    OS << " RenderNormal";
    break;
  case comments::InlineCommandComment::RenderBold:
    OS << " RenderBold";
    break;
  case comments::InlineCommandComment::RenderMonospaced:
    OS << " RenderMonospaced";
    break;
  case comments::InlineCommandComment::RenderEmphasized:
    OS << " RenderEmphasized";
    break;
  case comments::InlineCommandComment::RenderAnchor:
    OS << " RenderAnchor";
    break;
  }

  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void TextNodeDumper::visitHTMLStartTagComment(
    const comments::HTMLStartTagComment *C, const comments::FullComment *) {
  OS << " Name=\"" << C->getTagName() << "\"";
  if (C->getNumAttrs() != 0) {
    OS << " Attrs: ";
    for (unsigned i = 0, e = C->getNumAttrs(); i != e; ++i) {
      const comments::HTMLStartTagComment::Attribute &Attr = C->getAttr(i);
      OS << " \"" << Attr.Name << "=\"" << Attr.Value << "\"";
    }
  }
  if (C->isSelfClosing())
    OS << " SelfClosing";
}

void TextNodeDumper::visitHTMLEndTagComment(
    const comments::HTMLEndTagComment *C, const comments::FullComment *) {
  OS << " Name=\"" << C->getTagName() << "\"";
}

void TextNodeDumper::visitBlockCommandComment(
    const comments::BlockCommandComment *C, const comments::FullComment *) {
  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void TextNodeDumper::visitParamCommandComment(
    const comments::ParamCommandComment *C, const comments::FullComment *FC) {
  OS << " "
     << comments::ParamCommandComment::getDirectionAsString(C->getDirection());

  if (C->isDirectionExplicit())
    OS << " explicitly";
  else
    OS << " implicitly";

  if (C->hasParamName()) {
    if (C->isParamIndexValid())
      OS << " Param=\"" << C->getParamName(FC) << "\"";
    else
      OS << " Param=\"" << C->getParamNameAsWritten() << "\"";
  }

  if (C->isParamIndexValid() && !C->isVarArgParam())
    OS << " ParamIndex=" << C->getParamIndex();
}

void TextNodeDumper::visitTParamCommandComment(
    const comments::TParamCommandComment *C, const comments::FullComment *FC) {
  if (C->hasParamName()) {
    if (C->isPositionValid())
      OS << " Param=\"" << C->getParamName(FC) << "\"";
    else
      OS << " Param=\"" << C->getParamNameAsWritten() << "\"";
  }

  if (C->isPositionValid()) {
    OS << " Position=<";
    for (unsigned i = 0, e = C->getDepth(); i != e; ++i) {
      OS << C->getIndex(i);
      if (i != e - 1)
        OS << ", ";
    }
    OS << ">";
  }
}

void TextNodeDumper::visitVerbatimBlockComment(
    const comments::VerbatimBlockComment *C, const comments::FullComment *) {
  OS << " Name=\"" << getCommandName(C->getCommandID())
     << "\""
        " CloseName=\""
     << C->getCloseName() << "\"";
}

void TextNodeDumper::visitVerbatimBlockLineComment(
    const comments::VerbatimBlockLineComment *C,
    const comments::FullComment *) {
  OS << " Text=\"" << C->getText() << "\"";
}

void TextNodeDumper::visitVerbatimLineComment(
    const comments::VerbatimLineComment *C, const comments::FullComment *) {
  OS << " Text=\"" << C->getText() << "\"";
}

void TextNodeDumper::VisitNullTemplateArgument(const TemplateArgument &) {
  OS << " null";
}

void TextNodeDumper::VisitTypeTemplateArgument(const TemplateArgument &TA) {
  OS << " type";
  dumpType(TA.getAsType());
}

void TextNodeDumper::VisitDeclarationTemplateArgument(
    const TemplateArgument &TA) {
  OS << " decl";
  dumpDeclRef(TA.getAsDecl());
}

void TextNodeDumper::VisitNullPtrTemplateArgument(const TemplateArgument &) {
  OS << " nullptr";
}

void TextNodeDumper::VisitIntegralTemplateArgument(const TemplateArgument &TA) {
  OS << " integral " << TA.getAsIntegral();
}

void TextNodeDumper::VisitTemplateTemplateArgument(const TemplateArgument &TA) {
  OS << " template ";
  TA.getAsTemplate().dump(OS);
}

void TextNodeDumper::VisitTemplateExpansionTemplateArgument(
    const TemplateArgument &TA) {
  OS << " template expansion ";
  TA.getAsTemplateOrTemplatePattern().dump(OS);
}

void TextNodeDumper::VisitExpressionTemplateArgument(const TemplateArgument &) {
  OS << " expr";
}

void TextNodeDumper::VisitPackTemplateArgument(const TemplateArgument &) {
  OS << " pack";
}

static void dumpBasePath(raw_ostream &OS, const CastExpr *Node) {
  if (Node->path_empty())
    return;

  OS << " (";
  bool First = true;
  for (CastExpr::path_const_iterator I = Node->path_begin(),
                                     E = Node->path_end();
       I != E; ++I) {
    const CXXBaseSpecifier *Base = *I;
    if (!First)
      OS << " -> ";

    const auto *RD =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());

    if (Base->isVirtual())
      OS << "virtual ";
    OS << RD->getName();
    First = false;
  }

  OS << ')';
}

void TextNodeDumper::VisitIfStmt(const IfStmt *Node) {
  if (Node->hasInitStorage())
    OS << " has_init";
  if (Node->hasVarStorage())
    OS << " has_var";
  if (Node->hasElseStorage())
    OS << " has_else";
}

void TextNodeDumper::VisitSwitchStmt(const SwitchStmt *Node) {
  if (Node->hasInitStorage())
    OS << " has_init";
  if (Node->hasVarStorage())
    OS << " has_var";
}

void TextNodeDumper::VisitWhileStmt(const WhileStmt *Node) {
  if (Node->hasVarStorage())
    OS << " has_var";
}

void TextNodeDumper::VisitLabelStmt(const LabelStmt *Node) {
  OS << " '" << Node->getName() << "'";
}

void TextNodeDumper::VisitGotoStmt(const GotoStmt *Node) {
  OS << " '" << Node->getLabel()->getName() << "'";
  dumpPointer(Node->getLabel());
}

void TextNodeDumper::VisitCaseStmt(const CaseStmt *Node) {
  if (Node->caseStmtIsGNURange())
    OS << " gnu_range";
}

void TextNodeDumper::VisitConstantExpr(const ConstantExpr *Node) {
  if (Node->getResultAPValueKind() != APValue::None) {
    ColorScope Color(OS, ShowColors, ValueColor);
    OS << " ";
    Node->getAPValueResult().dump(OS);
  }
}

void TextNodeDumper::VisitCallExpr(const CallExpr *Node) {
  if (Node->usesADL())
    OS << " adl";
}

void TextNodeDumper::VisitCastExpr(const CastExpr *Node) {
  OS << " <";
  {
    ColorScope Color(OS, ShowColors, CastColor);
    OS << Node->getCastKindName();
  }
  dumpBasePath(OS, Node);
  OS << ">";
}

void TextNodeDumper::VisitImplicitCastExpr(const ImplicitCastExpr *Node) {
  VisitCastExpr(Node);
  if (Node->isPartOfExplicitCast())
    OS << " part_of_explicit_cast";
}

void TextNodeDumper::VisitDeclRefExpr(const DeclRefExpr *Node) {
  OS << " ";
  dumpBareDeclRef(Node->getDecl());
  if (Node->getDecl() != Node->getFoundDecl()) {
    OS << " (";
    dumpBareDeclRef(Node->getFoundDecl());
    OS << ")";
  }
  switch (Node->isNonOdrUse()) {
  case NOUR_None: break;
  case NOUR_Unevaluated: OS << " non_odr_use_unevaluated"; break;
  case NOUR_Constant: OS << " non_odr_use_constant"; break;
  case NOUR_Discarded: OS << " non_odr_use_discarded"; break;
  }
}

void TextNodeDumper::VisitUnresolvedLookupExpr(
    const UnresolvedLookupExpr *Node) {
  OS << " (";
  if (!Node->requiresADL())
    OS << "no ";
  OS << "ADL) = '" << Node->getName() << '\'';

  UnresolvedLookupExpr::decls_iterator I = Node->decls_begin(),
                                       E = Node->decls_end();
  if (I == E)
    OS << " empty";
  for (; I != E; ++I)
    dumpPointer(*I);
}

void TextNodeDumper::VisitObjCIvarRefExpr(const ObjCIvarRefExpr *Node) {
  {
    ColorScope Color(OS, ShowColors, DeclKindNameColor);
    OS << " " << Node->getDecl()->getDeclKindName() << "Decl";
  }
  OS << "='" << *Node->getDecl() << "'";
  dumpPointer(Node->getDecl());
  if (Node->isFreeIvar())
    OS << " isFreeIvar";
}

void TextNodeDumper::VisitPredefinedExpr(const PredefinedExpr *Node) {
  OS << " " << PredefinedExpr::getIdentKindName(Node->getIdentKind());
}

void TextNodeDumper::VisitCharacterLiteral(const CharacterLiteral *Node) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValue();
}

void TextNodeDumper::VisitIntegerLiteral(const IntegerLiteral *Node) {
  bool isSigned = Node->getType()->isSignedIntegerType();
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValue().toString(10, isSigned);
}

void TextNodeDumper::VisitFixedPointLiteral(const FixedPointLiteral *Node) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValueAsString(/*Radix=*/10);
}

void TextNodeDumper::VisitFloatingLiteral(const FloatingLiteral *Node) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " " << Node->getValueAsApproximateDouble();
}

void TextNodeDumper::VisitStringLiteral(const StringLiteral *Str) {
  ColorScope Color(OS, ShowColors, ValueColor);
  OS << " ";
  Str->outputString(OS);
}

void TextNodeDumper::VisitInitListExpr(const InitListExpr *ILE) {
  if (auto *Field = ILE->getInitializedFieldInUnion()) {
    OS << " field ";
    dumpBareDeclRef(Field);
  }
}

void TextNodeDumper::VisitGenericSelectionExpr(const GenericSelectionExpr *E) {
  if (E->isResultDependent())
    OS << " result_dependent";
}

void TextNodeDumper::VisitUnaryOperator(const UnaryOperator *Node) {
  OS << " " << (Node->isPostfix() ? "postfix" : "prefix") << " '"
     << UnaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
  if (!Node->canOverflow())
    OS << " cannot overflow";
}

void TextNodeDumper::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *Node) {
  switch (Node->getKind()) {
  case UETT_SizeOf:
    OS << " sizeof";
    break;
  case UETT_AlignOf:
    OS << " alignof";
    break;
  case UETT_VecStep:
    OS << " vec_step";
    break;
  case UETT_OpenMPRequiredSimdAlign:
    OS << " __builtin_omp_required_simd_align";
    break;
  case UETT_PreferredAlignOf:
    OS << " __alignof";
    break;
  }
  if (Node->isArgumentType())
    dumpType(Node->getArgumentType());
}

void TextNodeDumper::VisitMemberExpr(const MemberExpr *Node) {
  OS << " " << (Node->isArrow() ? "->" : ".") << *Node->getMemberDecl();
  dumpPointer(Node->getMemberDecl());
  switch (Node->isNonOdrUse()) {
  case NOUR_None: break;
  case NOUR_Unevaluated: OS << " non_odr_use_unevaluated"; break;
  case NOUR_Constant: OS << " non_odr_use_constant"; break;
  case NOUR_Discarded: OS << " non_odr_use_discarded"; break;
  }
}

void TextNodeDumper::VisitExtVectorElementExpr(
    const ExtVectorElementExpr *Node) {
  OS << " " << Node->getAccessor().getNameStart();
}

void TextNodeDumper::VisitBinaryOperator(const BinaryOperator *Node) {
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}

void TextNodeDumper::VisitCompoundAssignOperator(
    const CompoundAssignOperator *Node) {
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode())
     << "' ComputeLHSTy=";
  dumpBareType(Node->getComputationLHSType());
  OS << " ComputeResultTy=";
  dumpBareType(Node->getComputationResultType());
}

void TextNodeDumper::VisitAddrLabelExpr(const AddrLabelExpr *Node) {
  OS << " " << Node->getLabel()->getName();
  dumpPointer(Node->getLabel());
}

void TextNodeDumper::VisitCXXNamedCastExpr(const CXXNamedCastExpr *Node) {
  OS << " " << Node->getCastName() << "<"
     << Node->getTypeAsWritten().getAsString() << ">"
     << " <" << Node->getCastKindName();
  dumpBasePath(OS, Node);
  OS << ">";
}

void TextNodeDumper::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *Node) {
  OS << " " << (Node->getValue() ? "true" : "false");
}

void TextNodeDumper::VisitCXXThisExpr(const CXXThisExpr *Node) {
  if (Node->isImplicit())
    OS << " implicit";
  OS << " this";
}

void TextNodeDumper::VisitCXXFunctionalCastExpr(
    const CXXFunctionalCastExpr *Node) {
  OS << " functional cast to " << Node->getTypeAsWritten().getAsString() << " <"
     << Node->getCastKindName() << ">";
}

void TextNodeDumper::VisitCXXUnresolvedConstructExpr(
    const CXXUnresolvedConstructExpr *Node) {
  dumpType(Node->getTypeAsWritten());
  if (Node->isListInitialization())
    OS << " list";
}

void TextNodeDumper::VisitCXXConstructExpr(const CXXConstructExpr *Node) {
  CXXConstructorDecl *Ctor = Node->getConstructor();
  dumpType(Ctor->getType());
  if (Node->isElidable())
    OS << " elidable";
  if (Node->isListInitialization())
    OS << " list";
  if (Node->isStdInitListInitialization())
    OS << " std::initializer_list";
  if (Node->requiresZeroInitialization())
    OS << " zeroing";
}

void TextNodeDumper::VisitCXXBindTemporaryExpr(
    const CXXBindTemporaryExpr *Node) {
  OS << " (CXXTemporary";
  dumpPointer(Node);
  OS << ")";
}

void TextNodeDumper::VisitCXXNewExpr(const CXXNewExpr *Node) {
  if (Node->isGlobalNew())
    OS << " global";
  if (Node->isArray())
    OS << " array";
  if (Node->getOperatorNew()) {
    OS << ' ';
    dumpBareDeclRef(Node->getOperatorNew());
  }
  // We could dump the deallocation function used in case of error, but it's
  // usually not that interesting.
}

void TextNodeDumper::VisitCXXDeleteExpr(const CXXDeleteExpr *Node) {
  if (Node->isGlobalDelete())
    OS << " global";
  if (Node->isArrayForm())
    OS << " array";
  if (Node->getOperatorDelete()) {
    OS << ' ';
    dumpBareDeclRef(Node->getOperatorDelete());
  }
}

void TextNodeDumper::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *Node) {
  if (const ValueDecl *VD = Node->getExtendingDecl()) {
    OS << " extended by ";
    dumpBareDeclRef(VD);
  }
}

void TextNodeDumper::VisitExprWithCleanups(const ExprWithCleanups *Node) {
  for (unsigned i = 0, e = Node->getNumObjects(); i != e; ++i)
    dumpCleanupObject(Node->getObject(i));
}

void TextNodeDumper::VisitSizeOfPackExpr(const SizeOfPackExpr *Node) {
  dumpPointer(Node->getPack());
  dumpName(Node->getPack());
}

void TextNodeDumper::VisitCXXDependentScopeMemberExpr(
    const CXXDependentScopeMemberExpr *Node) {
  OS << " " << (Node->isArrow() ? "->" : ".") << Node->getMember();
}

void TextNodeDumper::VisitObjCMessageExpr(const ObjCMessageExpr *Node) {
  OS << " selector=";
  Node->getSelector().print(OS);
  switch (Node->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    break;

  case ObjCMessageExpr::Class:
    OS << " class=";
    dumpBareType(Node->getClassReceiver());
    break;

  case ObjCMessageExpr::SuperInstance:
    OS << " super (instance)";
    break;

  case ObjCMessageExpr::SuperClass:
    OS << " super (class)";
    break;
  }
}

void TextNodeDumper::VisitObjCBoxedExpr(const ObjCBoxedExpr *Node) {
  if (auto *BoxingMethod = Node->getBoxingMethod()) {
    OS << " selector=";
    BoxingMethod->getSelector().print(OS);
  }
}

void TextNodeDumper::VisitObjCAtCatchStmt(const ObjCAtCatchStmt *Node) {
  if (!Node->getCatchParamDecl())
    OS << " catch all";
}

void TextNodeDumper::VisitObjCEncodeExpr(const ObjCEncodeExpr *Node) {
  dumpType(Node->getEncodedType());
}

void TextNodeDumper::VisitObjCSelectorExpr(const ObjCSelectorExpr *Node) {
  OS << " ";
  Node->getSelector().print(OS);
}

void TextNodeDumper::VisitObjCProtocolExpr(const ObjCProtocolExpr *Node) {
  OS << ' ' << *Node->getProtocol();
}

void TextNodeDumper::VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *Node) {
  if (Node->isImplicitProperty()) {
    OS << " Kind=MethodRef Getter=\"";
    if (Node->getImplicitPropertyGetter())
      Node->getImplicitPropertyGetter()->getSelector().print(OS);
    else
      OS << "(null)";

    OS << "\" Setter=\"";
    if (ObjCMethodDecl *Setter = Node->getImplicitPropertySetter())
      Setter->getSelector().print(OS);
    else
      OS << "(null)";
    OS << "\"";
  } else {
    OS << " Kind=PropertyRef Property=\"" << *Node->getExplicitProperty()
       << '"';
  }

  if (Node->isSuperReceiver())
    OS << " super";

  OS << " Messaging=";
  if (Node->isMessagingGetter() && Node->isMessagingSetter())
    OS << "Getter&Setter";
  else if (Node->isMessagingGetter())
    OS << "Getter";
  else if (Node->isMessagingSetter())
    OS << "Setter";
}

void TextNodeDumper::VisitObjCSubscriptRefExpr(
    const ObjCSubscriptRefExpr *Node) {
  if (Node->isArraySubscriptRefExpr())
    OS << " Kind=ArraySubscript GetterForArray=\"";
  else
    OS << " Kind=DictionarySubscript GetterForDictionary=\"";
  if (Node->getAtIndexMethodDecl())
    Node->getAtIndexMethodDecl()->getSelector().print(OS);
  else
    OS << "(null)";

  if (Node->isArraySubscriptRefExpr())
    OS << "\" SetterForArray=\"";
  else
    OS << "\" SetterForDictionary=\"";
  if (Node->setAtIndexMethodDecl())
    Node->setAtIndexMethodDecl()->getSelector().print(OS);
  else
    OS << "(null)";
}

void TextNodeDumper::VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *Node) {
  OS << " " << (Node->getValue() ? "__objc_yes" : "__objc_no");
}

void TextNodeDumper::VisitOMPIteratorExpr(const OMPIteratorExpr *Node) {
  OS << " ";
  for (unsigned I = 0, E = Node->numOfIterators(); I < E; ++I) {
    Visit(Node->getIteratorDecl(I));
    OS << " = ";
    const OMPIteratorExpr::IteratorRange Range = Node->getIteratorRange(I);
    OS << " begin ";
    Visit(Range.Begin);
    OS << " end ";
    Visit(Range.End);
    if (Range.Step) {
      OS << " step ";
      Visit(Range.Step);
    }
  }
}

void TextNodeDumper::VisitRValueReferenceType(const ReferenceType *T) {
  if (T->isSpelledAsLValue())
    OS << " written as lvalue reference";
}

void TextNodeDumper::VisitArrayType(const ArrayType *T) {
  switch (T->getSizeModifier()) {
  case ArrayType::Normal:
    break;
  case ArrayType::Static:
    OS << " static";
    break;
  case ArrayType::Star:
    OS << " *";
    break;
  }
  OS << " " << T->getIndexTypeQualifiers().getAsString();
}

void TextNodeDumper::VisitConstantArrayType(const ConstantArrayType *T) {
  OS << " " << T->getSize();
  VisitArrayType(T);
}

void TextNodeDumper::VisitVariableArrayType(const VariableArrayType *T) {
  OS << " ";
  dumpSourceRange(T->getBracketsRange());
  VisitArrayType(T);
}

void TextNodeDumper::VisitDependentSizedArrayType(
    const DependentSizedArrayType *T) {
  VisitArrayType(T);
  OS << " ";
  dumpSourceRange(T->getBracketsRange());
}

void TextNodeDumper::VisitDependentSizedExtVectorType(
    const DependentSizedExtVectorType *T) {
  OS << " ";
  dumpLocation(T->getAttributeLoc());
}

void TextNodeDumper::VisitVectorType(const VectorType *T) {
  switch (T->getVectorKind()) {
  case VectorType::GenericVector:
    break;
  case VectorType::AltiVecVector:
    OS << " altivec";
    break;
  case VectorType::AltiVecPixel:
    OS << " altivec pixel";
    break;
  case VectorType::AltiVecBool:
    OS << " altivec bool";
    break;
  case VectorType::NeonVector:
    OS << " neon";
    break;
  case VectorType::NeonPolyVector:
    OS << " neon poly";
    break;
  }
  OS << " " << T->getNumElements();
}

void TextNodeDumper::VisitFunctionType(const FunctionType *T) {
  auto EI = T->getExtInfo();
  if (EI.getNoReturn())
    OS << " noreturn";
  if (EI.getProducesResult())
    OS << " produces_result";
  if (EI.getHasRegParm())
    OS << " regparm " << EI.getRegParm();
  OS << " " << FunctionType::getNameForCallConv(EI.getCC());
}

void TextNodeDumper::VisitFunctionProtoType(const FunctionProtoType *T) {
  auto EPI = T->getExtProtoInfo();
  if (EPI.HasTrailingReturn)
    OS << " trailing_return";
  if (T->isConst())
    OS << " const";
  if (T->isVolatile())
    OS << " volatile";
  if (T->isRestrict())
    OS << " restrict";
  if (T->getExtProtoInfo().Variadic)
    OS << " variadic";
  switch (EPI.RefQualifier) {
  case RQ_None:
    break;
  case RQ_LValue:
    OS << " &";
    break;
  case RQ_RValue:
    OS << " &&";
    break;
  }
  // FIXME: Exception specification.
  // FIXME: Consumed parameters.
  VisitFunctionType(T);
}

void TextNodeDumper::VisitUnresolvedUsingType(const UnresolvedUsingType *T) {
  dumpDeclRef(T->getDecl());
}

void TextNodeDumper::VisitTypedefType(const TypedefType *T) {
  dumpDeclRef(T->getDecl());
}

void TextNodeDumper::VisitUnaryTransformType(const UnaryTransformType *T) {
  switch (T->getUTTKind()) {
  case UnaryTransformType::EnumUnderlyingType:
    OS << " underlying_type";
    break;
  }
}

void TextNodeDumper::VisitTagType(const TagType *T) {
  dumpDeclRef(T->getDecl());
}

void TextNodeDumper::VisitTemplateTypeParmType(const TemplateTypeParmType *T) {
  OS << " depth " << T->getDepth() << " index " << T->getIndex();
  if (T->isParameterPack())
    OS << " pack";
  dumpDeclRef(T->getDecl());
}

void TextNodeDumper::VisitAutoType(const AutoType *T) {
  if (T->isDecltypeAuto())
    OS << " decltype(auto)";
  if (!T->isDeduced())
    OS << " undeduced";
  if (T->isConstrained()) {
    dumpDeclRef(T->getTypeConstraintConcept());
    for (const auto &Arg : T->getTypeConstraintArguments())
      VisitTemplateArgument(Arg);
  }
}

void TextNodeDumper::VisitTemplateSpecializationType(
    const TemplateSpecializationType *T) {
  if (T->isTypeAlias())
    OS << " alias";
  OS << " ";
  T->getTemplateName().dump(OS);
}

void TextNodeDumper::VisitInjectedClassNameType(
    const InjectedClassNameType *T) {
  dumpDeclRef(T->getDecl());
}

void TextNodeDumper::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  dumpDeclRef(T->getDecl());
}

void TextNodeDumper::VisitPackExpansionType(const PackExpansionType *T) {
  if (auto N = T->getNumExpansions())
    OS << " expansions " << *N;
}

void TextNodeDumper::VisitLabelDecl(const LabelDecl *D) { dumpName(D); }

void TextNodeDumper::VisitTypedefDecl(const TypedefDecl *D) {
  dumpName(D);
  dumpType(D->getUnderlyingType());
  if (D->isModulePrivate())
    OS << " __module_private__";
}

void TextNodeDumper::VisitEnumDecl(const EnumDecl *D) {
  if (D->isScoped()) {
    if (D->isScopedUsingClassTag())
      OS << " class";
    else
      OS << " struct";
  }
  dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isFixed())
    dumpType(D->getIntegerType());
}

void TextNodeDumper::VisitRecordDecl(const RecordDecl *D) {
  OS << ' ' << D->getKindName();
  dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isCompleteDefinition())
    OS << " definition";
}

void TextNodeDumper::VisitEnumConstantDecl(const EnumConstantDecl *D) {
  dumpName(D);
  dumpType(D->getType());
}

void TextNodeDumper::VisitIndirectFieldDecl(const IndirectFieldDecl *D) {
  dumpName(D);
  dumpType(D->getType());

  for (const auto *Child : D->chain())
    dumpDeclRef(Child);
}

void TextNodeDumper::VisitFunctionDecl(const FunctionDecl *D) {
  dumpName(D);
  dumpType(D->getType());

  StorageClass SC = D->getStorageClass();
  if (SC != SC_None)
    OS << ' ' << VarDecl::getStorageClassSpecifierString(SC);
  if (D->isInlineSpecified())
    OS << " inline";
  if (D->isVirtualAsWritten())
    OS << " virtual";
  if (D->isModulePrivate())
    OS << " __module_private__";

  if (D->isPure())
    OS << " pure";
  if (D->isDefaulted()) {
    OS << " default";
    if (D->isDeleted())
      OS << "_delete";
  }
  if (D->isDeletedAsWritten())
    OS << " delete";
  if (D->isTrivial())
    OS << " trivial";

  if (const auto *FPT = D->getType()->getAs<FunctionProtoType>()) {
    FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
    switch (EPI.ExceptionSpec.Type) {
    default:
      break;
    case EST_Unevaluated:
      OS << " noexcept-unevaluated " << EPI.ExceptionSpec.SourceDecl;
      break;
    case EST_Uninstantiated:
      OS << " noexcept-uninstantiated " << EPI.ExceptionSpec.SourceTemplate;
      break;
    }
  }

  if (const auto *MD = dyn_cast<CXXMethodDecl>(D)) {
    if (MD->size_overridden_methods() != 0) {
      auto dumpOverride = [=](const CXXMethodDecl *D) {
        SplitQualType T_split = D->getType().split();
        OS << D << " " << D->getParent()->getName()
           << "::" << D->getNameAsString() << " '"
           << QualType::getAsString(T_split, PrintPolicy) << "'";
      };

      AddChild([=] {
        auto Overrides = MD->overridden_methods();
        OS << "Overrides: [ ";
        dumpOverride(*Overrides.begin());
        for (const auto *Override :
             llvm::make_range(Overrides.begin() + 1, Overrides.end())) {
          OS << ", ";
          dumpOverride(Override);
        }
        OS << " ]";
      });
    }
  }

  // Since NumParams comes from the FunctionProtoType of the FunctionDecl and
  // the Params are set later, it is possible for a dump during debugging to
  // encounter a FunctionDecl that has been created but hasn't been assigned
  // ParmVarDecls yet.
  if (!D->param_empty() && !D->param_begin())
    OS << " <<<NULL params x " << D->getNumParams() << ">>>";
}

void TextNodeDumper::VisitLifetimeExtendedTemporaryDecl(
    const LifetimeExtendedTemporaryDecl *D) {
  OS << " extended by ";
  dumpBareDeclRef(D->getExtendingDecl());
  OS << " mangling ";
  {
    ColorScope Color(OS, ShowColors, ValueColor);
    OS << D->getManglingNumber();
  }
}

void TextNodeDumper::VisitFieldDecl(const FieldDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (D->isMutable())
    OS << " mutable";
  if (D->isModulePrivate())
    OS << " __module_private__";
}

void TextNodeDumper::VisitVarDecl(const VarDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  StorageClass SC = D->getStorageClass();
  if (SC != SC_None)
    OS << ' ' << VarDecl::getStorageClassSpecifierString(SC);
  switch (D->getTLSKind()) {
  case VarDecl::TLS_None:
    break;
  case VarDecl::TLS_Static:
    OS << " tls";
    break;
  case VarDecl::TLS_Dynamic:
    OS << " tls_dynamic";
    break;
  }
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isNRVOVariable())
    OS << " nrvo";
  if (D->isInline())
    OS << " inline";
  if (D->isConstexpr())
    OS << " constexpr";
  if (D->hasInit()) {
    switch (D->getInitStyle()) {
    case VarDecl::CInit:
      OS << " cinit";
      break;
    case VarDecl::CallInit:
      OS << " callinit";
      break;
    case VarDecl::ListInit:
      OS << " listinit";
      break;
    }
  }
  if (D->needsDestruction(D->getASTContext()))
    OS << " destroyed";
  if (D->isParameterPack())
    OS << " pack";
}

void TextNodeDumper::VisitBindingDecl(const BindingDecl *D) {
  dumpName(D);
  dumpType(D->getType());
}

void TextNodeDumper::VisitCapturedDecl(const CapturedDecl *D) {
  if (D->isNothrow())
    OS << " nothrow";
}

void TextNodeDumper::VisitImportDecl(const ImportDecl *D) {
  OS << ' ' << D->getImportedModule()->getFullModuleName();

  for (Decl *InitD :
       D->getASTContext().getModuleInitializers(D->getImportedModule()))
    dumpDeclRef(InitD, "initializer");
}

void TextNodeDumper::VisitPragmaCommentDecl(const PragmaCommentDecl *D) {
  OS << ' ';
  switch (D->getCommentKind()) {
  case PCK_Unknown:
    llvm_unreachable("unexpected pragma comment kind");
  case PCK_Compiler:
    OS << "compiler";
    break;
  case PCK_ExeStr:
    OS << "exestr";
    break;
  case PCK_Lib:
    OS << "lib";
    break;
  case PCK_Linker:
    OS << "linker";
    break;
  case PCK_User:
    OS << "user";
    break;
  }
  StringRef Arg = D->getArg();
  if (!Arg.empty())
    OS << " \"" << Arg << "\"";
}

void TextNodeDumper::VisitPragmaDetectMismatchDecl(
    const PragmaDetectMismatchDecl *D) {
  OS << " \"" << D->getName() << "\" \"" << D->getValue() << "\"";
}

void TextNodeDumper::VisitOMPExecutableDirective(
    const OMPExecutableDirective *D) {
  if (D->isStandaloneDirective())
    OS << " openmp_standalone_directive";
}

void TextNodeDumper::VisitOMPDeclareReductionDecl(
    const OMPDeclareReductionDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  OS << " combiner";
  dumpPointer(D->getCombiner());
  if (const auto *Initializer = D->getInitializer()) {
    OS << " initializer";
    dumpPointer(Initializer);
    switch (D->getInitializerKind()) {
    case OMPDeclareReductionDecl::DirectInit:
      OS << " omp_priv = ";
      break;
    case OMPDeclareReductionDecl::CopyInit:
      OS << " omp_priv ()";
      break;
    case OMPDeclareReductionDecl::CallInit:
      break;
    }
  }
}

void TextNodeDumper::VisitOMPRequiresDecl(const OMPRequiresDecl *D) {
  for (const auto *C : D->clauselists()) {
    AddChild([=] {
      if (!C) {
        ColorScope Color(OS, ShowColors, NullColor);
        OS << "<<<NULL>>> OMPClause";
        return;
      }
      {
        ColorScope Color(OS, ShowColors, AttrColor);
        StringRef ClauseName(
            llvm::omp::getOpenMPClauseName(C->getClauseKind()));
        OS << "OMP" << ClauseName.substr(/*Start=*/0, /*N=*/1).upper()
           << ClauseName.drop_front() << "Clause";
      }
      dumpPointer(C);
      dumpSourceRange(SourceRange(C->getBeginLoc(), C->getEndLoc()));
    });
  }
}

void TextNodeDumper::VisitOMPCapturedExprDecl(const OMPCapturedExprDecl *D) {
  dumpName(D);
  dumpType(D->getType());
}

void TextNodeDumper::VisitNamespaceDecl(const NamespaceDecl *D) {
  dumpName(D);
  if (D->isInline())
    OS << " inline";
  if (!D->isOriginalNamespace())
    dumpDeclRef(D->getOriginalNamespace(), "original");
}

void TextNodeDumper::VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
  OS << ' ';
  dumpBareDeclRef(D->getNominatedNamespace());
}

void TextNodeDumper::VisitNamespaceAliasDecl(const NamespaceAliasDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getAliasedNamespace());
}

void TextNodeDumper::VisitTypeAliasDecl(const TypeAliasDecl *D) {
  dumpName(D);
  dumpType(D->getUnderlyingType());
}

void TextNodeDumper::VisitTypeAliasTemplateDecl(
    const TypeAliasTemplateDecl *D) {
  dumpName(D);
}

void TextNodeDumper::VisitCXXRecordDecl(const CXXRecordDecl *D) {
  VisitRecordDecl(D);
  if (!D->isCompleteDefinition())
    return;

  AddChild([=] {
    {
      ColorScope Color(OS, ShowColors, DeclKindNameColor);
      OS << "DefinitionData";
    }
#define FLAG(fn, name)                                                         \
  if (D->fn())                                                                 \
    OS << " " #name;
    FLAG(isParsingBaseSpecifiers, parsing_base_specifiers);

    FLAG(isGenericLambda, generic);
    FLAG(isLambda, lambda);

    FLAG(isAnonymousStructOrUnion, is_anonymous);
    FLAG(canPassInRegisters, pass_in_registers);
    FLAG(isEmpty, empty);
    FLAG(isAggregate, aggregate);
    FLAG(isStandardLayout, standard_layout);
    FLAG(isTriviallyCopyable, trivially_copyable);
    FLAG(isPOD, pod);
    FLAG(isTrivial, trivial);
    FLAG(isPolymorphic, polymorphic);
    FLAG(isAbstract, abstract);
    FLAG(isLiteral, literal);

    FLAG(hasUserDeclaredConstructor, has_user_declared_ctor);
    FLAG(hasConstexprNonCopyMoveConstructor, has_constexpr_non_copy_move_ctor);
    FLAG(hasMutableFields, has_mutable_fields);
    FLAG(hasVariantMembers, has_variant_members);
    FLAG(allowConstDefaultInit, can_const_default_init);

    AddChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "DefaultConstructor";
      }
      FLAG(hasDefaultConstructor, exists);
      FLAG(hasTrivialDefaultConstructor, trivial);
      FLAG(hasNonTrivialDefaultConstructor, non_trivial);
      FLAG(hasUserProvidedDefaultConstructor, user_provided);
      FLAG(hasConstexprDefaultConstructor, constexpr);
      FLAG(needsImplicitDefaultConstructor, needs_implicit);
      FLAG(defaultedDefaultConstructorIsConstexpr, defaulted_is_constexpr);
    });

    AddChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "CopyConstructor";
      }
      FLAG(hasSimpleCopyConstructor, simple);
      FLAG(hasTrivialCopyConstructor, trivial);
      FLAG(hasNonTrivialCopyConstructor, non_trivial);
      FLAG(hasUserDeclaredCopyConstructor, user_declared);
      FLAG(hasCopyConstructorWithConstParam, has_const_param);
      FLAG(needsImplicitCopyConstructor, needs_implicit);
      FLAG(needsOverloadResolutionForCopyConstructor,
           needs_overload_resolution);
      if (!D->needsOverloadResolutionForCopyConstructor())
        FLAG(defaultedCopyConstructorIsDeleted, defaulted_is_deleted);
      FLAG(implicitCopyConstructorHasConstParam, implicit_has_const_param);
    });

    AddChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "MoveConstructor";
      }
      FLAG(hasMoveConstructor, exists);
      FLAG(hasSimpleMoveConstructor, simple);
      FLAG(hasTrivialMoveConstructor, trivial);
      FLAG(hasNonTrivialMoveConstructor, non_trivial);
      FLAG(hasUserDeclaredMoveConstructor, user_declared);
      FLAG(needsImplicitMoveConstructor, needs_implicit);
      FLAG(needsOverloadResolutionForMoveConstructor,
           needs_overload_resolution);
      if (!D->needsOverloadResolutionForMoveConstructor())
        FLAG(defaultedMoveConstructorIsDeleted, defaulted_is_deleted);
    });

    AddChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "CopyAssignment";
      }
      FLAG(hasSimpleCopyAssignment, simple);
      FLAG(hasTrivialCopyAssignment, trivial);
      FLAG(hasNonTrivialCopyAssignment, non_trivial);
      FLAG(hasCopyAssignmentWithConstParam, has_const_param);
      FLAG(hasUserDeclaredCopyAssignment, user_declared);
      FLAG(needsImplicitCopyAssignment, needs_implicit);
      FLAG(needsOverloadResolutionForCopyAssignment, needs_overload_resolution);
      FLAG(implicitCopyAssignmentHasConstParam, implicit_has_const_param);
    });

    AddChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "MoveAssignment";
      }
      FLAG(hasMoveAssignment, exists);
      FLAG(hasSimpleMoveAssignment, simple);
      FLAG(hasTrivialMoveAssignment, trivial);
      FLAG(hasNonTrivialMoveAssignment, non_trivial);
      FLAG(hasUserDeclaredMoveAssignment, user_declared);
      FLAG(needsImplicitMoveAssignment, needs_implicit);
      FLAG(needsOverloadResolutionForMoveAssignment, needs_overload_resolution);
    });

    AddChild([=] {
      {
        ColorScope Color(OS, ShowColors, DeclKindNameColor);
        OS << "Destructor";
      }
      FLAG(hasSimpleDestructor, simple);
      FLAG(hasIrrelevantDestructor, irrelevant);
      FLAG(hasTrivialDestructor, trivial);
      FLAG(hasNonTrivialDestructor, non_trivial);
      FLAG(hasUserDeclaredDestructor, user_declared);
      FLAG(hasConstexprDestructor, constexpr);
      FLAG(needsImplicitDestructor, needs_implicit);
      FLAG(needsOverloadResolutionForDestructor, needs_overload_resolution);
      if (!D->needsOverloadResolutionForDestructor())
        FLAG(defaultedDestructorIsDeleted, defaulted_is_deleted);
    });
  });

  for (const auto &I : D->bases()) {
    AddChild([=] {
      if (I.isVirtual())
        OS << "virtual ";
      dumpAccessSpecifier(I.getAccessSpecifier());
      dumpType(I.getType());
      if (I.isPackExpansion())
        OS << "...";
    });
  }
}

void TextNodeDumper::VisitFunctionTemplateDecl(const FunctionTemplateDecl *D) {
  dumpName(D);
}

void TextNodeDumper::VisitClassTemplateDecl(const ClassTemplateDecl *D) {
  dumpName(D);
}

void TextNodeDumper::VisitVarTemplateDecl(const VarTemplateDecl *D) {
  dumpName(D);
}

void TextNodeDumper::VisitBuiltinTemplateDecl(const BuiltinTemplateDecl *D) {
  dumpName(D);
}

void TextNodeDumper::VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *D) {
  if (const auto *TC = D->getTypeConstraint()) {
    OS << " ";
    dumpBareDeclRef(TC->getNamedConcept());
    if (TC->getNamedConcept() != TC->getFoundDecl()) {
      OS << " (";
      dumpBareDeclRef(TC->getFoundDecl());
      OS << ")";
    }
    Visit(TC->getImmediatelyDeclaredConstraint());
  } else if (D->wasDeclaredWithTypename())
    OS << " typename";
  else
    OS << " class";
  OS << " depth " << D->getDepth() << " index " << D->getIndex();
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
}

void TextNodeDumper::VisitNonTypeTemplateParmDecl(
    const NonTypeTemplateParmDecl *D) {
  dumpType(D->getType());
  OS << " depth " << D->getDepth() << " index " << D->getIndex();
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
}

void TextNodeDumper::VisitTemplateTemplateParmDecl(
    const TemplateTemplateParmDecl *D) {
  OS << " depth " << D->getDepth() << " index " << D->getIndex();
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
}

void TextNodeDumper::VisitUsingDecl(const UsingDecl *D) {
  OS << ' ';
  if (D->getQualifier())
    D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void TextNodeDumper::VisitUnresolvedUsingTypenameDecl(
    const UnresolvedUsingTypenameDecl *D) {
  OS << ' ';
  if (D->getQualifier())
    D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void TextNodeDumper::VisitUnresolvedUsingValueDecl(
    const UnresolvedUsingValueDecl *D) {
  OS << ' ';
  if (D->getQualifier())
    D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
  dumpType(D->getType());
}

void TextNodeDumper::VisitUsingShadowDecl(const UsingShadowDecl *D) {
  OS << ' ';
  dumpBareDeclRef(D->getTargetDecl());
}

void TextNodeDumper::VisitConstructorUsingShadowDecl(
    const ConstructorUsingShadowDecl *D) {
  if (D->constructsVirtualBase())
    OS << " virtual";

  AddChild([=] {
    OS << "target ";
    dumpBareDeclRef(D->getTargetDecl());
  });

  AddChild([=] {
    OS << "nominated ";
    dumpBareDeclRef(D->getNominatedBaseClass());
    OS << ' ';
    dumpBareDeclRef(D->getNominatedBaseClassShadowDecl());
  });

  AddChild([=] {
    OS << "constructed ";
    dumpBareDeclRef(D->getConstructedBaseClass());
    OS << ' ';
    dumpBareDeclRef(D->getConstructedBaseClassShadowDecl());
  });
}

void TextNodeDumper::VisitLinkageSpecDecl(const LinkageSpecDecl *D) {
  switch (D->getLanguage()) {
  case LinkageSpecDecl::lang_c:
    OS << " C";
    break;
  case LinkageSpecDecl::lang_cxx:
    OS << " C++";
    break;
  }
}

void TextNodeDumper::VisitAccessSpecDecl(const AccessSpecDecl *D) {
  OS << ' ';
  dumpAccessSpecifier(D->getAccess());
}

void TextNodeDumper::VisitFriendDecl(const FriendDecl *D) {
  if (TypeSourceInfo *T = D->getFriendType())
    dumpType(T->getType());
}

void TextNodeDumper::VisitObjCIvarDecl(const ObjCIvarDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (D->getSynthesize())
    OS << " synthesize";

  switch (D->getAccessControl()) {
  case ObjCIvarDecl::None:
    OS << " none";
    break;
  case ObjCIvarDecl::Private:
    OS << " private";
    break;
  case ObjCIvarDecl::Protected:
    OS << " protected";
    break;
  case ObjCIvarDecl::Public:
    OS << " public";
    break;
  case ObjCIvarDecl::Package:
    OS << " package";
    break;
  }
}

void TextNodeDumper::VisitObjCMethodDecl(const ObjCMethodDecl *D) {
  if (D->isInstanceMethod())
    OS << " -";
  else
    OS << " +";
  dumpName(D);
  dumpType(D->getReturnType());

  if (D->isVariadic())
    OS << " variadic";
}

void TextNodeDumper::VisitObjCTypeParamDecl(const ObjCTypeParamDecl *D) {
  dumpName(D);
  switch (D->getVariance()) {
  case ObjCTypeParamVariance::Invariant:
    break;

  case ObjCTypeParamVariance::Covariant:
    OS << " covariant";
    break;

  case ObjCTypeParamVariance::Contravariant:
    OS << " contravariant";
    break;
  }

  if (D->hasExplicitBound())
    OS << " bounded";
  dumpType(D->getUnderlyingType());
}

void TextNodeDumper::VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
  dumpDeclRef(D->getImplementation());
  for (const auto *P : D->protocols())
    dumpDeclRef(P);
}

void TextNodeDumper::VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
  dumpDeclRef(D->getCategoryDecl());
}

void TextNodeDumper::VisitObjCProtocolDecl(const ObjCProtocolDecl *D) {
  dumpName(D);

  for (const auto *Child : D->protocols())
    dumpDeclRef(Child);
}

void TextNodeDumper::VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getSuperClass(), "super");

  dumpDeclRef(D->getImplementation());
  for (const auto *Child : D->protocols())
    dumpDeclRef(Child);
}

void TextNodeDumper::VisitObjCImplementationDecl(
    const ObjCImplementationDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getSuperClass(), "super");
  dumpDeclRef(D->getClassInterface());
}

void TextNodeDumper::VisitObjCCompatibleAliasDecl(
    const ObjCCompatibleAliasDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
}

void TextNodeDumper::VisitObjCPropertyDecl(const ObjCPropertyDecl *D) {
  dumpName(D);
  dumpType(D->getType());

  if (D->getPropertyImplementation() == ObjCPropertyDecl::Required)
    OS << " required";
  else if (D->getPropertyImplementation() == ObjCPropertyDecl::Optional)
    OS << " optional";

  ObjCPropertyAttribute::Kind Attrs = D->getPropertyAttributes();
  if (Attrs != ObjCPropertyAttribute::kind_noattr) {
    if (Attrs & ObjCPropertyAttribute::kind_readonly)
      OS << " readonly";
    if (Attrs & ObjCPropertyAttribute::kind_assign)
      OS << " assign";
    if (Attrs & ObjCPropertyAttribute::kind_readwrite)
      OS << " readwrite";
    if (Attrs & ObjCPropertyAttribute::kind_retain)
      OS << " retain";
    if (Attrs & ObjCPropertyAttribute::kind_copy)
      OS << " copy";
    if (Attrs & ObjCPropertyAttribute::kind_nonatomic)
      OS << " nonatomic";
    if (Attrs & ObjCPropertyAttribute::kind_atomic)
      OS << " atomic";
    if (Attrs & ObjCPropertyAttribute::kind_weak)
      OS << " weak";
    if (Attrs & ObjCPropertyAttribute::kind_strong)
      OS << " strong";
    if (Attrs & ObjCPropertyAttribute::kind_unsafe_unretained)
      OS << " unsafe_unretained";
    if (Attrs & ObjCPropertyAttribute::kind_class)
      OS << " class";
    if (Attrs & ObjCPropertyAttribute::kind_direct)
      OS << " direct";
    if (Attrs & ObjCPropertyAttribute::kind_getter)
      dumpDeclRef(D->getGetterMethodDecl(), "getter");
    if (Attrs & ObjCPropertyAttribute::kind_setter)
      dumpDeclRef(D->getSetterMethodDecl(), "setter");
  }
}

void TextNodeDumper::VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D) {
  dumpName(D->getPropertyDecl());
  if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
    OS << " synthesize";
  else
    OS << " dynamic";
  dumpDeclRef(D->getPropertyDecl());
  dumpDeclRef(D->getPropertyIvarDecl());
}

void TextNodeDumper::VisitBlockDecl(const BlockDecl *D) {
  if (D->isVariadic())
    OS << " variadic";

  if (D->capturesCXXThis())
    OS << " captures_this";
}

void TextNodeDumper::VisitConceptDecl(const ConceptDecl *D) {
  dumpName(D);
}
