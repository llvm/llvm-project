//===---- StmtProfile.cpp - Profile implementation for Stmt ASTs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::Profile method, which builds a unique bit
// representation that identifies a statement/expression.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/ADT/FoldingSet.h"
using namespace clang;

namespace {
  class StmtProfiler : public ConstStmtVisitor<StmtProfiler> {
    llvm::FoldingSetNodeID &ID;
    const ASTContext &Context;
    bool Canonical;

  public:
    StmtProfiler(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                 bool Canonical)
      : ID(ID), Context(Context), Canonical(Canonical) { }

    void VisitStmt(const Stmt *S);

#define STMT(Node, Base) void Visit##Node(const Node *S);
#include "clang/AST/StmtNodes.inc"

    /// \brief Visit a declaration that is referenced within an expression
    /// or statement.
    void VisitDecl(const Decl *D);

    /// \brief Visit a type that is referenced within an expression or
    /// statement.
    void VisitType(QualType T);

    /// \brief Visit a name that occurs within an expression or statement.
    void VisitName(DeclarationName Name);

    /// \brief Visit a nested-name-specifier that occurs within an expression
    /// or statement.
    void VisitNestedNameSpecifier(NestedNameSpecifier *NNS);

    /// \brief Visit a template name that occurs within an expression or
    /// statement.
    void VisitTemplateName(TemplateName Name);

    /// \brief Visit template arguments that occur within an expression or
    /// statement.
    void VisitTemplateArguments(const TemplateArgumentLoc *Args,
                                unsigned NumArgs);

    /// \brief Visit a single template argument.
    void VisitTemplateArgument(const TemplateArgument &Arg);
  };
}

void StmtProfiler::VisitStmt(const Stmt *S) {
  ID.AddInteger(S->getStmtClass());
  for (Stmt::const_child_range C = S->children(); C; ++C) {
    if (*C)
      Visit(*C);
    else
      ID.AddInteger(0);
  }
}

void StmtProfiler::VisitDeclStmt(const DeclStmt *S) {
  VisitStmt(S);
  for (DeclStmt::const_decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D)
    VisitDecl(*D);
}

void StmtProfiler::VisitNullStmt(const NullStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitCompoundStmt(const CompoundStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitSwitchCase(const SwitchCase *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitCaseStmt(const CaseStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitDefaultStmt(const DefaultStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitLabelStmt(const LabelStmt *S) {
  VisitStmt(S);
  VisitDecl(S->getDecl());
}

void StmtProfiler::VisitAttributedStmt(const AttributedStmt *S) {
  VisitStmt(S);
  // TODO: maybe visit attributes?
}

void StmtProfiler::VisitIfStmt(const IfStmt *S) {
  VisitStmt(S);
  VisitDecl(S->getConditionVariable());
}

void StmtProfiler::VisitSwitchStmt(const SwitchStmt *S) {
  VisitStmt(S);
  VisitDecl(S->getConditionVariable());
}

void StmtProfiler::VisitWhileStmt(const WhileStmt *S) {
  VisitStmt(S);
  VisitDecl(S->getConditionVariable());
}

void StmtProfiler::VisitDoStmt(const DoStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitForStmt(const ForStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitGotoStmt(const GotoStmt *S) {
  VisitStmt(S);
  VisitDecl(S->getLabel());
}

void StmtProfiler::VisitIndirectGotoStmt(const IndirectGotoStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitContinueStmt(const ContinueStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitBreakStmt(const BreakStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitReturnStmt(const ReturnStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitGCCAsmStmt(const GCCAsmStmt *S) {
  VisitStmt(S);
  ID.AddBoolean(S->isVolatile());
  ID.AddBoolean(S->isSimple());
  VisitStringLiteral(S->getAsmString());
  ID.AddInteger(S->getNumOutputs());
  for (unsigned I = 0, N = S->getNumOutputs(); I != N; ++I) {
    ID.AddString(S->getOutputName(I));
    VisitStringLiteral(S->getOutputConstraintLiteral(I));
  }
  ID.AddInteger(S->getNumInputs());
  for (unsigned I = 0, N = S->getNumInputs(); I != N; ++I) {
    ID.AddString(S->getInputName(I));
    VisitStringLiteral(S->getInputConstraintLiteral(I));
  }
  ID.AddInteger(S->getNumClobbers());
  for (unsigned I = 0, N = S->getNumClobbers(); I != N; ++I)
    VisitStringLiteral(S->getClobberStringLiteral(I));
}

void StmtProfiler::VisitMSAsmStmt(const MSAsmStmt *S) {
  // FIXME: Implement MS style inline asm statement profiler.
  VisitStmt(S);
}

void StmtProfiler::VisitCXXCatchStmt(const CXXCatchStmt *S) {
  VisitStmt(S);
  VisitType(S->getCaughtType());
}

void StmtProfiler::VisitCXXTryStmt(const CXXTryStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitMSDependentExistsStmt(const MSDependentExistsStmt *S) {
  VisitStmt(S);
  ID.AddBoolean(S->isIfExists());
  VisitNestedNameSpecifier(S->getQualifierLoc().getNestedNameSpecifier());
  VisitName(S->getNameInfo().getName());
}

void StmtProfiler::VisitSEHTryStmt(const SEHTryStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitSEHFinallyStmt(const SEHFinallyStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitSEHExceptStmt(const SEHExceptStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCAtCatchStmt(const ObjCAtCatchStmt *S) {
  VisitStmt(S);
  ID.AddBoolean(S->hasEllipsis());
  if (S->getCatchParamDecl())
    VisitType(S->getCatchParamDecl()->getType());
}

void StmtProfiler::VisitObjCAtFinallyStmt(const ObjCAtFinallyStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCAtTryStmt(const ObjCAtTryStmt *S) {
  VisitStmt(S);
}

void
StmtProfiler::VisitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitObjCAtThrowStmt(const ObjCAtThrowStmt *S) {
  VisitStmt(S);
}

void
StmtProfiler::VisitObjCAutoreleasePoolStmt(const ObjCAutoreleasePoolStmt *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitExpr(const Expr *S) {
  VisitStmt(S);
}

void StmtProfiler::VisitDeclRefExpr(const DeclRefExpr *S) {
  VisitExpr(S);
  if (!Canonical)
    VisitNestedNameSpecifier(S->getQualifier());
  VisitDecl(S->getDecl());
  if (!Canonical)
    VisitTemplateArguments(S->getTemplateArgs(), S->getNumTemplateArgs());
}

void StmtProfiler::VisitPredefinedExpr(const PredefinedExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getIdentType());
}

void StmtProfiler::VisitIntegerLiteral(const IntegerLiteral *S) {
  VisitExpr(S);
  S->getValue().Profile(ID);
}

void StmtProfiler::VisitCharacterLiteral(const CharacterLiteral *S) {
  VisitExpr(S);
  ID.AddInteger(S->getKind());
  ID.AddInteger(S->getValue());
}

void StmtProfiler::VisitFloatingLiteral(const FloatingLiteral *S) {
  VisitExpr(S);
  S->getValue().Profile(ID);
  ID.AddBoolean(S->isExact());
}

void StmtProfiler::VisitImaginaryLiteral(const ImaginaryLiteral *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitStringLiteral(const StringLiteral *S) {
  VisitExpr(S);
  ID.AddString(S->getBytes());
  ID.AddInteger(S->getKind());
}

void StmtProfiler::VisitParenExpr(const ParenExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitParenListExpr(const ParenListExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitUnaryOperator(const UnaryOperator *S) {
  VisitExpr(S);
  ID.AddInteger(S->getOpcode());
}

void StmtProfiler::VisitOffsetOfExpr(const OffsetOfExpr *S) {
  VisitType(S->getTypeSourceInfo()->getType());
  unsigned n = S->getNumComponents();
  for (unsigned i = 0; i < n; ++i) {
    const OffsetOfExpr::OffsetOfNode& ON = S->getComponent(i);
    ID.AddInteger(ON.getKind());
    switch (ON.getKind()) {
    case OffsetOfExpr::OffsetOfNode::Array:
      // Expressions handled below.
      break;

    case OffsetOfExpr::OffsetOfNode::Field:
      VisitDecl(ON.getField());
      break;

    case OffsetOfExpr::OffsetOfNode::Identifier:
      ID.AddPointer(ON.getFieldName());
      break;
        
    case OffsetOfExpr::OffsetOfNode::Base:
      // These nodes are implicit, and therefore don't need profiling.
      break;
    }
  }
  
  VisitExpr(S);
}

void
StmtProfiler::VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getKind());
  if (S->isArgumentType())
    VisitType(S->getArgumentType());
}

void StmtProfiler::VisitArraySubscriptExpr(const ArraySubscriptExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCallExpr(const CallExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitMemberExpr(const MemberExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getMemberDecl());
  if (!Canonical)
    VisitNestedNameSpecifier(S->getQualifier());
  ID.AddBoolean(S->isArrow());
}

void StmtProfiler::VisitCompoundLiteralExpr(const CompoundLiteralExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isFileScope());
}

void StmtProfiler::VisitCastExpr(const CastExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitImplicitCastExpr(const ImplicitCastExpr *S) {
  VisitCastExpr(S);
  ID.AddInteger(S->getValueKind());
}

void StmtProfiler::VisitExplicitCastExpr(const ExplicitCastExpr *S) {
  VisitCastExpr(S);
  VisitType(S->getTypeAsWritten());
}

void StmtProfiler::VisitCStyleCastExpr(const CStyleCastExpr *S) {
  VisitExplicitCastExpr(S);
}

void StmtProfiler::VisitBinaryOperator(const BinaryOperator *S) {
  VisitExpr(S);
  ID.AddInteger(S->getOpcode());
}

void
StmtProfiler::VisitCompoundAssignOperator(const CompoundAssignOperator *S) {
  VisitBinaryOperator(S);
}

void StmtProfiler::VisitConditionalOperator(const ConditionalOperator *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitBinaryConditionalOperator(
    const BinaryConditionalOperator *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitAddrLabelExpr(const AddrLabelExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getLabel());
}

void StmtProfiler::VisitStmtExpr(const StmtExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitShuffleVectorExpr(const ShuffleVectorExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitChooseExpr(const ChooseExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitGNUNullExpr(const GNUNullExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitVAArgExpr(const VAArgExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitInitListExpr(const InitListExpr *S) {
  if (S->getSyntacticForm()) {
    VisitInitListExpr(S->getSyntacticForm());
    return;
  }

  VisitExpr(S);
}

void StmtProfiler::VisitDesignatedInitExpr(const DesignatedInitExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->usesGNUSyntax());
  for (DesignatedInitExpr::const_designators_iterator D =
         S->designators_begin(), DEnd = S->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      ID.AddInteger(0);
      VisitName(D->getFieldName());
      continue;
    }

    if (D->isArrayDesignator()) {
      ID.AddInteger(1);
    } else {
      assert(D->isArrayRangeDesignator());
      ID.AddInteger(2);
    }
    ID.AddInteger(D->getFirstExprIndex());
  }
}

void StmtProfiler::VisitImplicitValueInitExpr(const ImplicitValueInitExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitExtVectorElementExpr(const ExtVectorElementExpr *S) {
  VisitExpr(S);
  VisitName(&S->getAccessor());
}

void StmtProfiler::VisitBlockExpr(const BlockExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getBlockDecl());
}

void StmtProfiler::VisitGenericSelectionExpr(const GenericSelectionExpr *S) {
  VisitExpr(S);
  for (unsigned i = 0; i != S->getNumAssocs(); ++i) {
    QualType T = S->getAssocType(i);
    if (T.isNull())
      ID.AddPointer(0);
    else
      VisitType(T);
    VisitExpr(S->getAssocExpr(i));
  }
}

void StmtProfiler::VisitPseudoObjectExpr(const PseudoObjectExpr *S) {
  VisitExpr(S);
  for (PseudoObjectExpr::const_semantics_iterator
         i = S->semantics_begin(), e = S->semantics_end(); i != e; ++i)
    // Normally, we would not profile the source expressions of OVEs.
    if (const OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(*i))
      Visit(OVE->getSourceExpr());
}

void StmtProfiler::VisitAtomicExpr(const AtomicExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getOp());
}

static Stmt::StmtClass DecodeOperatorCall(const CXXOperatorCallExpr *S,
                                          UnaryOperatorKind &UnaryOp,
                                          BinaryOperatorKind &BinaryOp) {
  switch (S->getOperator()) {
  case OO_None:
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Arrow:
  case OO_Call:
  case OO_Conditional:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("Invalid operator call kind");
      
  case OO_Plus:
    if (S->getNumArgs() == 1) {
      UnaryOp = UO_Plus;
      return Stmt::UnaryOperatorClass;
    }
    
    BinaryOp = BO_Add;
    return Stmt::BinaryOperatorClass;
      
  case OO_Minus:
    if (S->getNumArgs() == 1) {
      UnaryOp = UO_Minus;
      return Stmt::UnaryOperatorClass;
    }
    
    BinaryOp = BO_Sub;
    return Stmt::BinaryOperatorClass;

  case OO_Star:
    if (S->getNumArgs() == 1) {
      UnaryOp = UO_Minus;
      return Stmt::UnaryOperatorClass;
    }
    
    BinaryOp = BO_Sub;
    return Stmt::BinaryOperatorClass;

  case OO_Slash:
    BinaryOp = BO_Div;
    return Stmt::BinaryOperatorClass;
      
  case OO_Percent:
    BinaryOp = BO_Rem;
    return Stmt::BinaryOperatorClass;

  case OO_Caret:
    BinaryOp = BO_Xor;
    return Stmt::BinaryOperatorClass;

  case OO_Amp:
    if (S->getNumArgs() == 1) {
      UnaryOp = UO_AddrOf;
      return Stmt::UnaryOperatorClass;
    }
    
    BinaryOp = BO_And;
    return Stmt::BinaryOperatorClass;
      
  case OO_Pipe:
    BinaryOp = BO_Or;
    return Stmt::BinaryOperatorClass;

  case OO_Tilde:
    UnaryOp = UO_Not;
    return Stmt::UnaryOperatorClass;

  case OO_Exclaim:
    UnaryOp = UO_LNot;
    return Stmt::UnaryOperatorClass;

  case OO_Equal:
    BinaryOp = BO_Assign;
    return Stmt::BinaryOperatorClass;

  case OO_Less:
    BinaryOp = BO_LT;
    return Stmt::BinaryOperatorClass;

  case OO_Greater:
    BinaryOp = BO_GT;
    return Stmt::BinaryOperatorClass;
      
  case OO_PlusEqual:
    BinaryOp = BO_AddAssign;
    return Stmt::CompoundAssignOperatorClass;

  case OO_MinusEqual:
    BinaryOp = BO_SubAssign;
    return Stmt::CompoundAssignOperatorClass;

  case OO_StarEqual:
    BinaryOp = BO_MulAssign;
    return Stmt::CompoundAssignOperatorClass;

  case OO_SlashEqual:
    BinaryOp = BO_DivAssign;
    return Stmt::CompoundAssignOperatorClass;

  case OO_PercentEqual:
    BinaryOp = BO_RemAssign;
    return Stmt::CompoundAssignOperatorClass;

  case OO_CaretEqual:
    BinaryOp = BO_XorAssign;
    return Stmt::CompoundAssignOperatorClass;
    
  case OO_AmpEqual:
    BinaryOp = BO_AndAssign;
    return Stmt::CompoundAssignOperatorClass;
    
  case OO_PipeEqual:
    BinaryOp = BO_OrAssign;
    return Stmt::CompoundAssignOperatorClass;
      
  case OO_LessLess:
    BinaryOp = BO_Shl;
    return Stmt::BinaryOperatorClass;
    
  case OO_GreaterGreater:
    BinaryOp = BO_Shr;
    return Stmt::BinaryOperatorClass;

  case OO_LessLessEqual:
    BinaryOp = BO_ShlAssign;
    return Stmt::CompoundAssignOperatorClass;
    
  case OO_GreaterGreaterEqual:
    BinaryOp = BO_ShrAssign;
    return Stmt::CompoundAssignOperatorClass;

  case OO_EqualEqual:
    BinaryOp = BO_EQ;
    return Stmt::BinaryOperatorClass;
    
  case OO_ExclaimEqual:
    BinaryOp = BO_NE;
    return Stmt::BinaryOperatorClass;
      
  case OO_LessEqual:
    BinaryOp = BO_LE;
    return Stmt::BinaryOperatorClass;
    
  case OO_GreaterEqual:
    BinaryOp = BO_GE;
    return Stmt::BinaryOperatorClass;
      
  case OO_AmpAmp:
    BinaryOp = BO_LAnd;
    return Stmt::BinaryOperatorClass;
    
  case OO_PipePipe:
    BinaryOp = BO_LOr;
    return Stmt::BinaryOperatorClass;

  case OO_PlusPlus:
    UnaryOp = S->getNumArgs() == 1? UO_PreInc 
                                  : UO_PostInc;
    return Stmt::UnaryOperatorClass;

  case OO_MinusMinus:
    UnaryOp = S->getNumArgs() == 1? UO_PreDec
                                  : UO_PostDec;
    return Stmt::UnaryOperatorClass;

  case OO_Comma:
    BinaryOp = BO_Comma;
    return Stmt::BinaryOperatorClass;


  case OO_ArrowStar:
    BinaryOp = BO_PtrMemI;
    return Stmt::BinaryOperatorClass;
      
  case OO_Subscript:
    return Stmt::ArraySubscriptExprClass;
  }
  
  llvm_unreachable("Invalid overloaded operator expression");
}
                               

void StmtProfiler::VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *S) {
  if (S->isTypeDependent()) {
    // Type-dependent operator calls are profiled like their underlying
    // syntactic operator.
    UnaryOperatorKind UnaryOp = UO_Extension;
    BinaryOperatorKind BinaryOp = BO_Comma;
    Stmt::StmtClass SC = DecodeOperatorCall(S, UnaryOp, BinaryOp);
    
    ID.AddInteger(SC);
    for (unsigned I = 0, N = S->getNumArgs(); I != N; ++I)
      Visit(S->getArg(I));
    if (SC == Stmt::UnaryOperatorClass)
      ID.AddInteger(UnaryOp);
    else if (SC == Stmt::BinaryOperatorClass || 
             SC == Stmt::CompoundAssignOperatorClass)
      ID.AddInteger(BinaryOp);
    else
      assert(SC == Stmt::ArraySubscriptExprClass);
                    
    return;
  }
  
  VisitCallExpr(S);
  ID.AddInteger(S->getOperator());
}

void StmtProfiler::VisitCXXMemberCallExpr(const CXXMemberCallExpr *S) {
  VisitCallExpr(S);
}

void StmtProfiler::VisitCUDAKernelCallExpr(const CUDAKernelCallExpr *S) {
  VisitCallExpr(S);
}

void StmtProfiler::VisitAsTypeExpr(const AsTypeExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXNamedCastExpr(const CXXNamedCastExpr *S) {
  VisitExplicitCastExpr(S);
}

void StmtProfiler::VisitCXXStaticCastExpr(const CXXStaticCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void StmtProfiler::VisitCXXDynamicCastExpr(const CXXDynamicCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void
StmtProfiler::VisitCXXReinterpretCastExpr(const CXXReinterpretCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void StmtProfiler::VisitCXXConstCastExpr(const CXXConstCastExpr *S) {
  VisitCXXNamedCastExpr(S);
}

void StmtProfiler::VisitUserDefinedLiteral(const UserDefinedLiteral *S) {
  VisitCallExpr(S);
}

void StmtProfiler::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->getValue());
}

void StmtProfiler::VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXTypeidExpr(const CXXTypeidExpr *S) {
  VisitExpr(S);
  if (S->isTypeOperand())
    VisitType(S->getTypeOperand());
}

void StmtProfiler::VisitCXXUuidofExpr(const CXXUuidofExpr *S) {
  VisitExpr(S);
  if (S->isTypeOperand())
    VisitType(S->getTypeOperand());
}

void StmtProfiler::VisitCXXThisExpr(const CXXThisExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isImplicit());
}

void StmtProfiler::VisitCXXThrowExpr(const CXXThrowExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getParam());
}

void StmtProfiler::VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *S) {
  VisitExpr(S);
  VisitDecl(
         const_cast<CXXDestructorDecl *>(S->getTemporary()->getDestructor()));
}

void StmtProfiler::VisitCXXConstructExpr(const CXXConstructExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getConstructor());
  ID.AddBoolean(S->isElidable());
}

void StmtProfiler::VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *S) {
  VisitExplicitCastExpr(S);
}

void
StmtProfiler::VisitCXXTemporaryObjectExpr(const CXXTemporaryObjectExpr *S) {
  VisitCXXConstructExpr(S);
}

void
StmtProfiler::VisitLambdaExpr(const LambdaExpr *S) {
  VisitExpr(S);
  for (LambdaExpr::capture_iterator C = S->explicit_capture_begin(),
                                 CEnd = S->explicit_capture_end();
       C != CEnd; ++C) {
    ID.AddInteger(C->getCaptureKind());
    if (C->capturesVariable()) {
      VisitDecl(C->getCapturedVar());
      ID.AddBoolean(C->isPackExpansion());
    }
  }
  // Note: If we actually needed to be able to match lambda
  // expressions, we would have to consider parameters and return type
  // here, among other things.
  VisitStmt(S->getBody());
}

void
StmtProfiler::VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXDeleteExpr(const CXXDeleteExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isGlobalDelete());
  ID.AddBoolean(S->isArrayForm());
  VisitDecl(S->getOperatorDelete());
}


void StmtProfiler::VisitCXXNewExpr(const CXXNewExpr *S) {
  VisitExpr(S);
  VisitType(S->getAllocatedType());
  VisitDecl(S->getOperatorNew());
  VisitDecl(S->getOperatorDelete());
  ID.AddBoolean(S->isArray());
  ID.AddInteger(S->getNumPlacementArgs());
  ID.AddBoolean(S->isGlobalNew());
  ID.AddBoolean(S->isParenTypeId());
  ID.AddInteger(S->getInitializationStyle());
}

void
StmtProfiler::VisitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isArrow());
  VisitNestedNameSpecifier(S->getQualifier());
  VisitType(S->getDestroyedType());
}

void StmtProfiler::VisitOverloadExpr(const OverloadExpr *S) {
  VisitExpr(S);
  VisitNestedNameSpecifier(S->getQualifier());
  VisitName(S->getName());
  ID.AddBoolean(S->hasExplicitTemplateArgs());
  if (S->hasExplicitTemplateArgs())
    VisitTemplateArguments(S->getExplicitTemplateArgs().getTemplateArgs(),
                           S->getExplicitTemplateArgs().NumTemplateArgs);
}

void
StmtProfiler::VisitUnresolvedLookupExpr(const UnresolvedLookupExpr *S) {
  VisitOverloadExpr(S);
}

void StmtProfiler::VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getTrait());
  VisitType(S->getQueriedType());
}

void StmtProfiler::VisitBinaryTypeTraitExpr(const BinaryTypeTraitExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getTrait());
  VisitType(S->getLhsType());
  VisitType(S->getRhsType());
}

void StmtProfiler::VisitTypeTraitExpr(const TypeTraitExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getTrait());
  ID.AddInteger(S->getNumArgs());
  for (unsigned I = 0, N = S->getNumArgs(); I != N; ++I)
    VisitType(S->getArg(I)->getType());
}

void StmtProfiler::VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getTrait());
  VisitType(S->getQueriedType());
}

void StmtProfiler::VisitExpressionTraitExpr(const ExpressionTraitExpr *S) {
  VisitExpr(S);
  ID.AddInteger(S->getTrait());
  VisitExpr(S->getQueriedExpression());
}

void StmtProfiler::VisitDependentScopeDeclRefExpr(
    const DependentScopeDeclRefExpr *S) {
  VisitExpr(S);
  VisitName(S->getDeclName());
  VisitNestedNameSpecifier(S->getQualifier());
  ID.AddBoolean(S->hasExplicitTemplateArgs());
  if (S->hasExplicitTemplateArgs())
    VisitTemplateArguments(S->getTemplateArgs(), S->getNumTemplateArgs());
}

void StmtProfiler::VisitExprWithCleanups(const ExprWithCleanups *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitCXXUnresolvedConstructExpr(
    const CXXUnresolvedConstructExpr *S) {
  VisitExpr(S);
  VisitType(S->getTypeAsWritten());
}

void StmtProfiler::VisitCXXDependentScopeMemberExpr(
    const CXXDependentScopeMemberExpr *S) {
  ID.AddBoolean(S->isImplicitAccess());
  if (!S->isImplicitAccess()) {
    VisitExpr(S);
    ID.AddBoolean(S->isArrow());
  }
  VisitNestedNameSpecifier(S->getQualifier());
  VisitName(S->getMember());
  ID.AddBoolean(S->hasExplicitTemplateArgs());
  if (S->hasExplicitTemplateArgs())
    VisitTemplateArguments(S->getTemplateArgs(), S->getNumTemplateArgs());
}

void StmtProfiler::VisitUnresolvedMemberExpr(const UnresolvedMemberExpr *S) {
  ID.AddBoolean(S->isImplicitAccess());
  if (!S->isImplicitAccess()) {
    VisitExpr(S);
    ID.AddBoolean(S->isArrow());
  }
  VisitNestedNameSpecifier(S->getQualifier());
  VisitName(S->getMemberName());
  ID.AddBoolean(S->hasExplicitTemplateArgs());
  if (S->hasExplicitTemplateArgs())
    VisitTemplateArguments(S->getTemplateArgs(), S->getNumTemplateArgs());
}

void StmtProfiler::VisitCXXNoexceptExpr(const CXXNoexceptExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitPackExpansionExpr(const PackExpansionExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitSizeOfPackExpr(const SizeOfPackExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getPack());
}

void StmtProfiler::VisitSubstNonTypeTemplateParmPackExpr(
    const SubstNonTypeTemplateParmPackExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getParameterPack());
  VisitTemplateArgument(S->getArgumentPack());
}

void StmtProfiler::VisitSubstNonTypeTemplateParmExpr(
    const SubstNonTypeTemplateParmExpr *E) {
  // Profile exactly as the replacement expression.
  Visit(E->getReplacement());
}

void StmtProfiler::VisitFunctionParmPackExpr(const FunctionParmPackExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getParameterPack());
  ID.AddInteger(S->getNumExpansions());
  for (FunctionParmPackExpr::iterator I = S->begin(), E = S->end(); I != E; ++I)
    VisitDecl(*I);
}

void StmtProfiler::VisitMaterializeTemporaryExpr(
                                           const MaterializeTemporaryExpr *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitOpaqueValueExpr(const OpaqueValueExpr *E) {
  VisitExpr(E);  
}

void StmtProfiler::VisitObjCStringLiteral(const ObjCStringLiteral *S) {
  VisitExpr(S);
}

void StmtProfiler::VisitObjCBoxedExpr(const ObjCBoxedExpr *E) {
  VisitExpr(E);
}

void StmtProfiler::VisitObjCArrayLiteral(const ObjCArrayLiteral *E) {
  VisitExpr(E);
}

void StmtProfiler::VisitObjCDictionaryLiteral(const ObjCDictionaryLiteral *E) {
  VisitExpr(E);
}

void StmtProfiler::VisitObjCEncodeExpr(const ObjCEncodeExpr *S) {
  VisitExpr(S);
  VisitType(S->getEncodedType());
}

void StmtProfiler::VisitObjCSelectorExpr(const ObjCSelectorExpr *S) {
  VisitExpr(S);
  VisitName(S->getSelector());
}

void StmtProfiler::VisitObjCProtocolExpr(const ObjCProtocolExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getProtocol());
}

void StmtProfiler::VisitObjCIvarRefExpr(const ObjCIvarRefExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getDecl());
  ID.AddBoolean(S->isArrow());
  ID.AddBoolean(S->isFreeIvar());
}

void StmtProfiler::VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *S) {
  VisitExpr(S);
  if (S->isImplicitProperty()) {
    VisitDecl(S->getImplicitPropertyGetter());
    VisitDecl(S->getImplicitPropertySetter());
  } else {
    VisitDecl(S->getExplicitProperty());
  }
  if (S->isSuperReceiver()) {
    ID.AddBoolean(S->isSuperReceiver());
    VisitType(S->getSuperReceiverType());
  }
}

void StmtProfiler::VisitObjCSubscriptRefExpr(const ObjCSubscriptRefExpr *S) {
  VisitExpr(S);
  VisitDecl(S->getAtIndexMethodDecl());
  VisitDecl(S->setAtIndexMethodDecl());
}

void StmtProfiler::VisitObjCMessageExpr(const ObjCMessageExpr *S) {
  VisitExpr(S);
  VisitName(S->getSelector());
  VisitDecl(S->getMethodDecl());
}

void StmtProfiler::VisitObjCIsaExpr(const ObjCIsaExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->isArrow());
}

void StmtProfiler::VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->getValue());
}

void StmtProfiler::VisitObjCIndirectCopyRestoreExpr(
    const ObjCIndirectCopyRestoreExpr *S) {
  VisitExpr(S);
  ID.AddBoolean(S->shouldCopy());
}

void StmtProfiler::VisitObjCBridgedCastExpr(const ObjCBridgedCastExpr *S) {
  VisitExplicitCastExpr(S);
  ID.AddBoolean(S->getBridgeKind());
}

void StmtProfiler::VisitDecl(const Decl *D) {
  ID.AddInteger(D? D->getKind() : 0);

  if (Canonical && D) {
    if (const NonTypeTemplateParmDecl *NTTP =
          dyn_cast<NonTypeTemplateParmDecl>(D)) {
      ID.AddInteger(NTTP->getDepth());
      ID.AddInteger(NTTP->getIndex());
      ID.AddBoolean(NTTP->isParameterPack());
      VisitType(NTTP->getType());
      return;
    }

    if (const ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
      // The Itanium C++ ABI uses the type, scope depth, and scope
      // index of a parameter when mangling expressions that involve
      // function parameters, so we will use the parameter's type for
      // establishing function parameter identity. That way, our
      // definition of "equivalent" (per C++ [temp.over.link]) is at
      // least as strong as the definition of "equivalent" used for
      // name mangling.
      VisitType(Parm->getType());
      ID.AddInteger(Parm->getFunctionScopeDepth());
      ID.AddInteger(Parm->getFunctionScopeIndex());
      return;
    }

    if (const TemplateTypeParmDecl *TTP =
          dyn_cast<TemplateTypeParmDecl>(D)) {
      ID.AddInteger(TTP->getDepth());
      ID.AddInteger(TTP->getIndex());
      ID.AddBoolean(TTP->isParameterPack());
      return;
    }

    if (const TemplateTemplateParmDecl *TTP =
          dyn_cast<TemplateTemplateParmDecl>(D)) {
      ID.AddInteger(TTP->getDepth());
      ID.AddInteger(TTP->getIndex());
      ID.AddBoolean(TTP->isParameterPack());
      return;
    }
  }

  ID.AddPointer(D? D->getCanonicalDecl() : 0);
}

void StmtProfiler::VisitType(QualType T) {
  if (Canonical)
    T = Context.getCanonicalType(T);

  ID.AddPointer(T.getAsOpaquePtr());
}

void StmtProfiler::VisitName(DeclarationName Name) {
  ID.AddPointer(Name.getAsOpaquePtr());
}

void StmtProfiler::VisitNestedNameSpecifier(NestedNameSpecifier *NNS) {
  if (Canonical)
    NNS = Context.getCanonicalNestedNameSpecifier(NNS);
  ID.AddPointer(NNS);
}

void StmtProfiler::VisitTemplateName(TemplateName Name) {
  if (Canonical)
    Name = Context.getCanonicalTemplateName(Name);

  Name.Profile(ID);
}

void StmtProfiler::VisitTemplateArguments(const TemplateArgumentLoc *Args,
                                          unsigned NumArgs) {
  ID.AddInteger(NumArgs);
  for (unsigned I = 0; I != NumArgs; ++I)
    VisitTemplateArgument(Args[I].getArgument());
}

void StmtProfiler::VisitTemplateArgument(const TemplateArgument &Arg) {
  // Mostly repetitive with TemplateArgument::Profile!
  ID.AddInteger(Arg.getKind());
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    break;

  case TemplateArgument::Type:
    VisitType(Arg.getAsType());
    break;

  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    VisitTemplateName(Arg.getAsTemplateOrTemplatePattern());
    break;
      
  case TemplateArgument::Declaration:
    VisitDecl(Arg.getAsDecl());
    break;

  case TemplateArgument::NullPtr:
    VisitType(Arg.getNullPtrType());
    break;

  case TemplateArgument::Integral:
    Arg.getAsIntegral().Profile(ID);
    VisitType(Arg.getIntegralType());
    break;

  case TemplateArgument::Expression:
    Visit(Arg.getAsExpr());
    break;

  case TemplateArgument::Pack:
    const TemplateArgument *Pack = Arg.pack_begin();
    for (unsigned i = 0, e = Arg.pack_size(); i != e; ++i)
      VisitTemplateArgument(Pack[i]);
    break;
  }
}

void Stmt::Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                   bool Canonical) const {
  StmtProfiler Profiler(ID, Context, Canonical);
  Profiler.Visit(this);
}
