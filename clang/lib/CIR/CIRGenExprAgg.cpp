//===--- CIRGenExprAgg.cpp - Emit CIR Code from Aggregate Expressions -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Aggregate Expr nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/StmtVisitor.h"

using namespace cir;
using namespace clang;

namespace {
class AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CIRGenFunction &CGF;
  AggValueSlot Dest;
  // bool IsResultUnused;

  AggValueSlot EnsureSlot(QualType T) {
    assert(!Dest.isIgnored() && "ignored slots NYI");
    return Dest;
  }

public:
  AggExprEmitter(CIRGenFunction &cgf, AggValueSlot Dest, bool IsResultUnused)
      : CGF{cgf}, Dest(Dest)
  // ,IsResultUnused(IsResultUnused)
  {}

  //===--------------------------------------------------------------------===//
  //                             Visitor Methods
  //===--------------------------------------------------------------------===//

  void Visit(Expr *E) {
    if (CGF.getDebugInfo()) {
      llvm_unreachable("NYI");
    }
    StmtVisitor<AggExprEmitter>::Visit(E);
  }

  void VisitStmt(Stmt *S) { llvm_unreachable("NYI"); }
  void VisitParenExpr(ParenExpr *PE) { llvm_unreachable("NYI"); }
  void VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    llvm_unreachable("NYI");
  }
  void VisitCoawaitExpr(CoawaitExpr *E) { llvm_unreachable("NYI"); }
  void VisitCoyieldExpr(CoyieldExpr *E) { llvm_unreachable("NYI"); }
  void VisitUnaryCoawait(UnaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitUnaryExtension(UnaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitConstantExpr(ConstantExpr *E) { llvm_unreachable("NYI"); }

  // l-values
  void VisitDeclRefExpr(DeclRefExpr *E) { llvm_unreachable("NYI"); }
  void VisitMemberExpr(MemberExpr *E) { llvm_unreachable("NYI"); }
  void VisitUnaryDeref(UnaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitStringLiteral(StringLiteral *E) { llvm_unreachable("NYI"); }
  void VisitCompoundLIteralExpr(CompoundLiteralExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitPredefinedExpr(const PredefinedExpr *E) { llvm_unreachable("NYI"); }

  // Operators.
  void VisitCastExpr(CastExpr *E) { llvm_unreachable("NYI"); }
  void VisitCallExpr(const CallExpr *E) { llvm_unreachable("NYI"); }
  void VisitStmtExpr(const StmtExpr *E) { llvm_unreachable("NYI"); }
  void VisitBinaryOperator(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitPointerToDataMemberBinaryOperator(const BinaryOperator *E) {
    llvm_unreachable("NYI");
  }
  void VisitBinAssign(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitBinComma(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitBinCmp(const BinaryOperator *E) { llvm_unreachable("NYI"); }
  void VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *E) {
    llvm_unreachable("NYI");
  }

  void VisitObjCMessageExpr(ObjCMessageExpr *E) { llvm_unreachable("NYI"); }
  void VisitObjCIVarRefExpr(ObjCIvarRefExpr *E) { llvm_unreachable("NYI"); }

  void VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    llvm_unreachable("NYI");
  }
  void VisitChooseExpr(const ChooseExpr *E) { llvm_unreachable("NYI"); }
  void VisitInitListExpr(InitListExpr *E) { llvm_unreachable("NYI"); }
  void VisitArrayInitLoopExpr(const ArrayInitLoopExpr *E,
                              llvm::Value *outerBegin = nullptr) {
    llvm_unreachable("NYI");
  }
  void VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitNoInitExpr(NoInitExpr *E) { llvm_unreachable("NYI"); }
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) { llvm_unreachable("NYI"); }
  void VisitXCXDefaultInitExpr(CXXDefaultInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitCXXConstructExpr(const CXXConstructExpr *E);
  void VisitCXXInheritedCtorInitExpr(const CXXInheritedCtorInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitLambdaExpr(LambdaExpr *E) { llvm_unreachable("NYI"); }
  void VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitExprWithCleanups(ExprWithCleanups *E);
  void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  void VisitCXXTypeidExpr(CXXTypeidExpr *E) { llvm_unreachable("NYI"); }
  void VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E);
  void VisitOpaqueValueExpr(OpaqueValueExpr *E) { llvm_unreachable("NYI"); }

  void VisitPseudoObjectExpr(PseudoObjectExpr *E) { llvm_unreachable("NYI"); }

  void VisitVAArgExpr(VAArgExpr *E) { llvm_unreachable("NYI"); }

  void EmitInitializationToLValue(Expr *E, LValue Address) {
    llvm_unreachable("NYI");
  }
  void EmitNullInitializationToLValue(LValue Address) {
    llvm_unreachable("NYI");
  }
  // case Expr::ChoseExprClass:
  void VisitCXXThrowExpr(const CXXThrowExpr *E) { llvm_unreachable("NYI"); }
  void VisitAtomicExpr(AtomicExpr *E) { llvm_unreachable("NYI"); }
};
} // namespace

//===----------------------------------------------------------------------===//
//                             Visitor Methods
//===----------------------------------------------------------------------===//

void AggExprEmitter::VisitMaterializeTemporaryExpr(
    MaterializeTemporaryExpr *E) {
  Visit(E->getSubExpr());
}

void AggExprEmitter::VisitCXXConstructExpr(const CXXConstructExpr *E) {
  AggValueSlot Slot = EnsureSlot(E->getType());
  CGF.buildCXXConstructExpr(E, Slot);
}

void AggExprEmitter::VisitExprWithCleanups(ExprWithCleanups *E) {
  if (UnimplementedFeature::cleanups())
    llvm_unreachable("NYI");
  Visit(E->getSubExpr());
}

/// CheckAggExprForMemSetUse - If the initializer is large and has a lot of
/// zeros in it, emit a memset and avoid storing the individual zeros.
static void CheckAggExprForMemSetUse(AggValueSlot &Slot, const Expr *E,
                                     CIRGenFunction &CGF) {
  // If the slot is arleady known to be zeroed, nothing to do. Don't mess with
  // volatile stores.
  if (Slot.isZeroed() || Slot.isVolatile() || !Slot.getAddress().isValid())
    return;

  // C++ objects with a user-declared constructor don't need zero'ing.
  if (CGF.getLangOpts().CPlusPlus)
    if (const auto *RT = CGF.getContext()
                             .getBaseElementType(E->getType())
                             ->getAs<RecordType>()) {
      const auto *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasUserDeclaredConstructor())
        return;
    }

  llvm_unreachable("NYI");
}

void CIRGenFunction::buildAggExpr(const Expr *E, AggValueSlot Slot) {
  assert(E && CIRGenFunction::hasAggregateEvaluationKind(E->getType()) &&
         "Invalid aggregate expression to emit");
  assert((Slot.getAddress().isValid() || Slot.isIgnored()) &&
         "slot has bits but no address");

  // Optimize the slot if possible.
  CheckAggExprForMemSetUse(Slot, E, *this);

  AggExprEmitter(*this, Slot, Slot.isIgnored()).Visit(const_cast<Expr *>(E));
}
