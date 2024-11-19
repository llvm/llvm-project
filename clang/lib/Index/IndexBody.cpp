//===- IndexBody.cpp - Indexing statements --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"
#include "clang/AST/ASTConcept.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"

using namespace clang;
using namespace clang::index;

namespace {

class BodyIndexer : public DynamicRecursiveASTVisitor {
  IndexingContext &IndexCtx;
  const NamedDecl *Parent;
  const DeclContext *ParentDC;
  SmallVector<Stmt*, 16> StmtStack;

  Stmt *getParentStmt() const {
    return StmtStack.size() < 2 ? nullptr : StmtStack.end()[-2];
  }
public:
  BodyIndexer(IndexingContext &indexCtx, const NamedDecl *Parent,
              const DeclContext *DC)
      : IndexCtx(indexCtx), Parent(Parent), ParentDC(DC) {
    ShouldWalkTypesOfTypeLocs = false;
  }

  bool dataTraverseStmtPre(Stmt *S) override {
    StmtStack.push_back(S);
    return true;
  }

  bool dataTraverseStmtPost(Stmt *S) override {
    assert(StmtStack.back() == S);
    StmtStack.pop_back();
    return true;
  }

  bool TraverseTypeLoc(TypeLoc TL) override {
    IndexCtx.indexTypeLoc(TL, Parent, ParentDC);
    return true;
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) override {
    IndexCtx.indexNestedNameSpecifierLoc(NNS, Parent, ParentDC);
    return true;
  }

  SymbolRoleSet getRolesForRef(const Expr *E,
                               SmallVectorImpl<SymbolRelation> &Relations) {
    SymbolRoleSet Roles{};
    assert(!StmtStack.empty() && E == StmtStack.back());
    if (StmtStack.size() == 1)
      return Roles;
    auto It = StmtStack.end()-2;
    while (isa<CastExpr>(*It) || isa<ParenExpr>(*It)) {
      if (auto ICE = dyn_cast<ImplicitCastExpr>(*It)) {
        if (ICE->getCastKind() == CK_LValueToRValue)
          Roles |= (unsigned)(unsigned)SymbolRole::Read;
      }
      if (It == StmtStack.begin())
        break;
      --It;
    }
    const Stmt *Parent = *It;

    if (auto BO = dyn_cast<BinaryOperator>(Parent)) {
      if (BO->getOpcode() == BO_Assign) {
        if (BO->getLHS()->IgnoreParenCasts() == E)
          Roles |= (unsigned)SymbolRole::Write;
      } else if (auto CA = dyn_cast<CompoundAssignOperator>(Parent)) {
        if (CA->getLHS()->IgnoreParenCasts() == E) {
          Roles |= (unsigned)SymbolRole::Read;
          Roles |= (unsigned)SymbolRole::Write;
        }
      }
    } else if (auto UO = dyn_cast<UnaryOperator>(Parent)) {
      if (UO->isIncrementDecrementOp()) {
        Roles |= (unsigned)SymbolRole::Read;
        Roles |= (unsigned)SymbolRole::Write;
      } else if (UO->getOpcode() == UO_AddrOf) {
        Roles |= (unsigned)SymbolRole::AddressOf;
      }

    } else if (auto CE = dyn_cast<CallExpr>(Parent)) {
      if (CE->getCallee()->IgnoreParenCasts() == E) {
        addCallRole(Roles, Relations);
        if (auto *ME = dyn_cast<MemberExpr>(E)) {
          if (auto *CXXMD = dyn_cast_or_null<CXXMethodDecl>(ME->getMemberDecl()))
            if (CXXMD->isVirtual() && !ME->hasQualifier()) {
              Roles |= (unsigned)SymbolRole::Dynamic;
              auto BaseTy = ME->getBase()->IgnoreImpCasts()->getType();
              if (!BaseTy.isNull())
                if (auto *CXXRD = BaseTy->getPointeeCXXRecordDecl())
                  Relations.emplace_back((unsigned)SymbolRole::RelationReceivedBy,
                                         CXXRD);
            }
        }
      } else if (auto CXXOp = dyn_cast<CXXOperatorCallExpr>(CE)) {
        if (CXXOp->getNumArgs() > 0 && CXXOp->getArg(0)->IgnoreParenCasts() == E) {
          OverloadedOperatorKind Op = CXXOp->getOperator();
          if (Op == OO_Equal) {
            Roles |= (unsigned)SymbolRole::Write;
          } else if ((Op >= OO_PlusEqual && Op <= OO_PipeEqual) ||
                     Op == OO_LessLessEqual || Op == OO_GreaterGreaterEqual ||
                     Op == OO_PlusPlus || Op == OO_MinusMinus) {
            Roles |= (unsigned)SymbolRole::Read;
            Roles |= (unsigned)SymbolRole::Write;
          } else if (Op == OO_Amp) {
            Roles |= (unsigned)SymbolRole::AddressOf;
          }
        }
      }
    }

    return Roles;
  }

  void addCallRole(SymbolRoleSet &Roles,
                   SmallVectorImpl<SymbolRelation> &Relations) {
    Roles |= (unsigned)SymbolRole::Call;
    if (auto *FD = dyn_cast<FunctionDecl>(ParentDC))
      Relations.emplace_back((unsigned)SymbolRole::RelationCalledBy, FD);
    else if (auto *MD = dyn_cast<ObjCMethodDecl>(ParentDC))
      Relations.emplace_back((unsigned)SymbolRole::RelationCalledBy, MD);
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) override {
    SmallVector<SymbolRelation, 4> Relations;
    SymbolRoleSet Roles = getRolesForRef(E, Relations);
    return IndexCtx.handleReference(E->getDecl(), E->getLocation(),
                                    Parent, ParentDC, Roles, Relations, E);
  }

  bool VisitGotoStmt(GotoStmt *S) override {
    return IndexCtx.handleReference(S->getLabel(), S->getLabelLoc(), Parent,
                                    ParentDC);
  }

  bool VisitLabelStmt(LabelStmt *S) override {
    if (IndexCtx.shouldIndexFunctionLocalSymbols())
      return IndexCtx.handleDecl(S->getDecl());
    return true;
  }

  bool VisitMemberExpr(MemberExpr *E) override {
    SourceLocation Loc = E->getMemberLoc();
    if (Loc.isInvalid())
      Loc = E->getBeginLoc();
    SmallVector<SymbolRelation, 4> Relations;
    SymbolRoleSet Roles = getRolesForRef(E, Relations);
    return IndexCtx.handleReference(E->getMemberDecl(), Loc,
                                    Parent, ParentDC, Roles, Relations, E);
  }

  bool indexDependentReference(
      const Expr *E, const Type *T, const DeclarationNameInfo &NameInfo,
      llvm::function_ref<bool(const NamedDecl *ND)> Filter) {
    if (!T)
      return true;
    const TemplateSpecializationType *TST =
        T->getAs<TemplateSpecializationType>();
    if (!TST)
      return true;
    TemplateName TN = TST->getTemplateName();
    const ClassTemplateDecl *TD =
        dyn_cast_or_null<ClassTemplateDecl>(TN.getAsTemplateDecl());
    if (!TD)
      return true;
    CXXRecordDecl *RD = TD->getTemplatedDecl();
    if (!RD->hasDefinition())
      return true;
    RD = RD->getDefinition();
    std::vector<const NamedDecl *> Symbols =
        RD->lookupDependentName(NameInfo.getName(), Filter);
    // FIXME: Improve overload handling.
    if (Symbols.size() != 1)
      return true;
    SourceLocation Loc = NameInfo.getLoc();
    if (Loc.isInvalid())
      Loc = E->getBeginLoc();
    SmallVector<SymbolRelation, 4> Relations;
    SymbolRoleSet Roles = getRolesForRef(E, Relations);
    return IndexCtx.handleReference(Symbols[0], Loc, Parent, ParentDC, Roles,
                                    Relations, E);
  }

  bool
  VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E) override {
    const DeclarationNameInfo &Info = E->getMemberNameInfo();
    return indexDependentReference(
        E, E->getBaseType().getTypePtrOrNull(), Info,
        [](const NamedDecl *D) { return D->isCXXInstanceMember(); });
  }

  bool VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) override {
    const DeclarationNameInfo &Info = E->getNameInfo();
    const NestedNameSpecifier *NNS = E->getQualifier();
    return indexDependentReference(
        E, NNS->getAsType(), Info,
        [](const NamedDecl *D) { return !D->isCXXInstanceMember(); });
  }

  bool VisitDesignatedInitExpr(DesignatedInitExpr *E) override {
    for (DesignatedInitExpr::Designator &D : llvm::reverse(E->designators())) {
      if (D.isFieldDesignator()) {
        if (const FieldDecl *FD = D.getFieldDecl()) {
          return IndexCtx.handleReference(FD, D.getFieldLoc(), Parent,
                                          ParentDC, SymbolRoleSet(), {}, E);
        }
      }
    }
    return true;
  }

  bool VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) override {
    SmallVector<SymbolRelation, 4> Relations;
    SymbolRoleSet Roles = getRolesForRef(E, Relations);
    return IndexCtx.handleReference(E->getDecl(), E->getLocation(),
                                    Parent, ParentDC, Roles, Relations, E);
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) override {
    auto isDynamic = [](const ObjCMessageExpr *MsgE)->bool {
      if (MsgE->getReceiverKind() != ObjCMessageExpr::Instance)
        return false;
      if (auto *RecE = dyn_cast<ObjCMessageExpr>(
              MsgE->getInstanceReceiver()->IgnoreParenCasts())) {
        if (RecE->getMethodFamily() == OMF_alloc)
          return false;
      }
      return true;
    };

    if (ObjCMethodDecl *MD = E->getMethodDecl()) {
      SymbolRoleSet Roles{};
      SmallVector<SymbolRelation, 2> Relations;
      addCallRole(Roles, Relations);
      Stmt *Containing = getParentStmt();

      auto IsImplicitProperty = [](const PseudoObjectExpr *POE) -> bool {
        const auto *E = POE->getSyntacticForm();
        if (const auto *BinOp = dyn_cast<BinaryOperator>(E))
          E = BinOp->getLHS();
        const auto *PRE = dyn_cast<ObjCPropertyRefExpr>(E);
        if (!PRE)
          return false;
        if (PRE->isExplicitProperty())
          return false;
        if (const ObjCMethodDecl *Getter = PRE->getImplicitPropertyGetter()) {
          // Class properties that are explicitly defined using @property
          // declarations are represented implicitly as there is no ivar for
          // class properties.
          if (Getter->isClassMethod() &&
              Getter->getCanonicalDecl()->findPropertyDecl())
            return false;
        }
        return true;
      };
      bool IsPropCall = isa_and_nonnull<PseudoObjectExpr>(Containing);
      // Implicit property message sends are not 'implicit'.
      if ((E->isImplicit() || IsPropCall) &&
          !(IsPropCall &&
            IsImplicitProperty(cast<PseudoObjectExpr>(Containing))))
        Roles |= (unsigned)SymbolRole::Implicit;

      if (isDynamic(E)) {
        Roles |= (unsigned)SymbolRole::Dynamic;

        auto addReceivers = [&](const ObjCObjectType *Ty) {
          if (!Ty)
            return;
          if (const auto *clsD = Ty->getInterface()) {
            Relations.emplace_back((unsigned)SymbolRole::RelationReceivedBy,
                                   clsD);
          }
          for (const auto *protD : Ty->quals()) {
            Relations.emplace_back((unsigned)SymbolRole::RelationReceivedBy,
                                   protD);
          }
        };
        QualType recT = E->getReceiverType();
        if (const auto *Ptr = recT->getAs<ObjCObjectPointerType>())
          addReceivers(Ptr->getObjectType());
        else
          addReceivers(recT->getAs<ObjCObjectType>());
      }

      return IndexCtx.handleReference(MD, E->getSelectorStartLoc(),
                                      Parent, ParentDC, Roles, Relations, E);
    }
    return true;
  }

  bool VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) override {
    if (E->isExplicitProperty()) {
      SmallVector<SymbolRelation, 2> Relations;
      SymbolRoleSet Roles = getRolesForRef(E, Relations);
      return IndexCtx.handleReference(E->getExplicitProperty(), E->getLocation(),
                                      Parent, ParentDC, Roles, Relations, E);
    } else if (const ObjCMethodDecl *Getter = E->getImplicitPropertyGetter()) {
      // Class properties that are explicitly defined using @property
      // declarations are represented implicitly as there is no ivar for class
      // properties.
      if (Getter->isClassMethod()) {
        if (const auto *PD = Getter->getCanonicalDecl()->findPropertyDecl()) {
          SmallVector<SymbolRelation, 2> Relations;
          SymbolRoleSet Roles = getRolesForRef(E, Relations);
          return IndexCtx.handleReference(PD, E->getLocation(), Parent,
                                          ParentDC, Roles, Relations, E);
        }
      }
    }

    // No need to do a handleReference for the objc method, because there will
    // be a message expr as part of PseudoObjectExpr.
    return true;
  }

  bool VisitMSPropertyRefExpr(MSPropertyRefExpr *E) override {
    return IndexCtx.handleReference(E->getPropertyDecl(), E->getMemberLoc(),
                                    Parent, ParentDC, SymbolRoleSet(), {}, E);
  }

  bool VisitObjCProtocolExpr(ObjCProtocolExpr *E) override {
    return IndexCtx.handleReference(E->getProtocol(), E->getProtocolIdLoc(),
                                    Parent, ParentDC, SymbolRoleSet(), {}, E);
  }

  bool passObjCLiteralMethodCall(const ObjCMethodDecl *MD, const Expr *E) {
    SymbolRoleSet Roles{};
    SmallVector<SymbolRelation, 2> Relations;
    addCallRole(Roles, Relations);
    Roles |= (unsigned)SymbolRole::Implicit;
    return IndexCtx.handleReference(MD, E->getBeginLoc(), Parent, ParentDC,
                                    Roles, Relations, E);
  }

  bool VisitObjCBoxedExpr(ObjCBoxedExpr *E) override {
    if (ObjCMethodDecl *MD = E->getBoxingMethod()) {
      return passObjCLiteralMethodCall(MD, E);
    }
    return true;
  }

  bool VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) override {
    if (ObjCMethodDecl *MD = E->getDictWithObjectsMethod()) {
      return passObjCLiteralMethodCall(MD, E);
    }
    return true;
  }

  bool VisitObjCArrayLiteral(ObjCArrayLiteral *E) override {
    if (ObjCMethodDecl *MD = E->getArrayWithObjectsMethod()) {
      return passObjCLiteralMethodCall(MD, E);
    }
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *E) override {
    SymbolRoleSet Roles{};
    SmallVector<SymbolRelation, 2> Relations;
    addCallRole(Roles, Relations);
    return IndexCtx.handleReference(E->getConstructor(), E->getLocation(),
                                    Parent, ParentDC, Roles, Relations, E);
  }

  bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr *E) override {
    if (E->getOperatorLoc().isInvalid())
      return true; // implicit.
    return DynamicRecursiveASTVisitor::TraverseCXXOperatorCallExpr(E);
  }

  bool VisitDeclStmt(DeclStmt *S) override {
    if (IndexCtx.shouldIndexFunctionLocalSymbols()) {
      IndexCtx.indexDeclGroupRef(S->getDeclGroup());
      return true;
    }

    DeclGroupRef DG = S->getDeclGroup();
    for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I) {
      const Decl *D = *I;
      if (!D)
        continue;
      if (!isFunctionLocalSymbol(D))
        IndexCtx.indexTopLevelDecl(D);
    }

    return true;
  }

  bool TraverseLambdaCapture(LambdaExpr *LE, const LambdaCapture *C,
                             Expr *Init) override {
    if (C->capturesThis() || C->capturesVLAType())
      return true;

    if (!DynamicRecursiveASTVisitor::TraverseStmt(Init))
      return false;

    if (C->capturesVariable() && IndexCtx.shouldIndexFunctionLocalSymbols())
      return IndexCtx.handleReference(C->getCapturedVar(), C->getLocation(),
                                      Parent, ParentDC, SymbolRoleSet());

    return true;
  }

  // RecursiveASTVisitor visits both syntactic and semantic forms, duplicating
  // the things that we visit. Make sure to only visit the semantic form.
  // Also visit things that are in the syntactic form but not the semantic one,
  // for example the indices in DesignatedInitExprs.
  bool TraverseInitListExpr(InitListExpr *S) override {
    auto visitForm = [&](InitListExpr *Form) {
      for (Stmt *SubStmt : Form->children()) {
        if (!TraverseStmt(SubStmt))
          return false;
      }
      return true;
    };

    auto visitSyntacticDesignatedInitExpr = [&](DesignatedInitExpr *E) -> bool {
      for (DesignatedInitExpr::Designator &D : llvm::reverse(E->designators())) {
        if (D.isFieldDesignator()) {
          if (const FieldDecl *FD = D.getFieldDecl()) {
            return IndexCtx.handleReference(FD, D.getFieldLoc(), Parent,
                                            ParentDC, SymbolRoleSet(),
                                            /*Relations=*/{}, E);
          }
        }
      }
      return true;
    };

    InitListExpr *SemaForm = S->isSemanticForm() ? S : S->getSemanticForm();
    InitListExpr *SyntaxForm = S->isSemanticForm() ? S->getSyntacticForm() : S;

    if (SemaForm) {
      // Visit things present in syntactic form but not the semantic form.
      if (SyntaxForm) {
        for (Expr *init : SyntaxForm->inits()) {
          if (auto *DIE = dyn_cast<DesignatedInitExpr>(init))
            visitSyntacticDesignatedInitExpr(DIE);
        }
      }
      return visitForm(SemaForm);
    }

    // No semantic, try the syntactic.
    if (SyntaxForm) {
      return visitForm(SyntaxForm);
    }

    return true;
  }

  bool VisitOffsetOfExpr(OffsetOfExpr *S) override {
    for (unsigned I = 0, E = S->getNumComponents(); I != E; ++I) {
      const OffsetOfNode &Component = S->getComponent(I);
      if (Component.getKind() == OffsetOfNode::Field)
        IndexCtx.handleReference(Component.getField(), Component.getEndLoc(),
                                 Parent, ParentDC, SymbolRoleSet(), {});
      // FIXME: Try to resolve dependent field references.
    }
    return true;
  }

  bool VisitParmVarDecl(ParmVarDecl *D) override {
    // Index the parameters of lambda expression and requires expression.
    if (IndexCtx.shouldIndexFunctionLocalSymbols()) {
      const auto *DC = D->getDeclContext();
      if (DC && (isLambdaCallOperator(DC) || isa<RequiresExprBodyDecl>(DC)))
        IndexCtx.handleDecl(D);
    }
    return true;
  }

  bool VisitOverloadExpr(OverloadExpr *E) override {
    SmallVector<SymbolRelation, 4> Relations;
    SymbolRoleSet Roles = getRolesForRef(E, Relations);
    for (auto *D : E->decls())
      IndexCtx.handleReference(D, E->getNameLoc(), Parent, ParentDC, Roles,
                               Relations, E);
    return true;
  }

  bool VisitConceptSpecializationExpr(ConceptSpecializationExpr *R) override {
    IndexCtx.handleReference(R->getNamedConcept(), R->getConceptNameLoc(),
                             Parent, ParentDC);
    return true;
  }

  bool TraverseTypeConstraint(const TypeConstraint *C) override {
    IndexCtx.handleReference(C->getNamedConcept(), C->getConceptNameLoc(),
                             Parent, ParentDC);
    return DynamicRecursiveASTVisitor::TraverseTypeConstraint(C);
  }
};

} // anonymous namespace

void IndexingContext::indexBody(const Stmt *S, const NamedDecl *Parent,
                                const DeclContext *DC) {
  if (!S)
    return;

  if (!DC)
    DC = Parent->getLexicalDeclContext();
  BodyIndexer(*this, Parent, DC).TraverseStmt(const_cast<Stmt*>(S));
}
