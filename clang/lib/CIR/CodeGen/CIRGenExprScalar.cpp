//===--- CIRGenExprScalar.cpp - Emit CIR Code for Scalar Exprs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with scalar CIR types as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRDataLayout.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

namespace {

struct BinOpInfo {
  mlir::Value LHS;
  mlir::Value RHS;
  SourceRange Loc;
  QualType Ty;                   // Computation Type.
  BinaryOperator::Opcode Opcode; // Opcode of BinOp to perform
  FPOptions FPFeatures;
  const Expr *E; // Entire expr, for error unsupported.  May not be binop.

  /// Check if the binop computes a division or a remainder.
  bool isDivremOp() const {
    return Opcode == BO_Div || Opcode == BO_Rem || Opcode == BO_DivAssign ||
           Opcode == BO_RemAssign;
  }

  /// Check if the binop can result in integer overflow.
  bool mayHaveIntegerOverflow() const {
    // Without constant input, we can't rule out overflow.
    auto LHSCI = dyn_cast<mlir::cir::ConstantOp>(LHS.getDefiningOp());
    auto RHSCI = dyn_cast<mlir::cir::ConstantOp>(RHS.getDefiningOp());
    if (!LHSCI || !RHSCI)
      return true;

    llvm::APInt Result;
    assert(!UnimplementedFeature::mayHaveIntegerOverflow());
    llvm_unreachable("NYI");
    return false;
  }

  /// Check if at least one operand is a fixed point type. In such cases,
  /// this operation did not follow usual arithmetic conversion and both
  /// operands might not be of the same type.
  bool isFixedPointOp() const {
    // We cannot simply check the result type since comparison operations
    // return an int.
    if (const auto *BinOp = llvm::dyn_cast<BinaryOperator>(E)) {
      QualType LHSType = BinOp->getLHS()->getType();
      QualType RHSType = BinOp->getRHS()->getType();
      return LHSType->isFixedPointType() || RHSType->isFixedPointType();
    }
    if (const auto *UnOp = llvm::dyn_cast<UnaryOperator>(E))
      return UnOp->getSubExpr()->getType()->isFixedPointType();
    return false;
  }
};

static bool PromotionIsPotentiallyEligibleForImplicitIntegerConversionCheck(
    QualType SrcType, QualType DstType) {
  return SrcType->isIntegerType() && DstType->isIntegerType();
}

class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
  CIRGenFunction &CGF;
  CIRGenBuilderTy &Builder;
  bool IgnoreResultAssign;

public:
  ScalarExprEmitter(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                    bool ira = false)
      : CGF(cgf), Builder(builder), IgnoreResultAssign(ira) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  bool TestAndClearIgnoreResultAssign() {
    bool I = IgnoreResultAssign;
    IgnoreResultAssign = false;
    return I;
  }

  mlir::Type ConvertType(QualType T) { return CGF.ConvertType(T); }
  LValue buildLValue(const Expr *E) { return CGF.buildLValue(E); }
  LValue buildCheckedLValue(const Expr *E, CIRGenFunction::TypeCheckKind TCK) {
    return CGF.buildCheckedLValue(E, TCK);
  }

  /// Emit a value that corresponds to null for the given type.
  mlir::Value buildNullValue(QualType Ty, mlir::Location loc);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *E) {
    return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(E);
  }

  mlir::Value VisitStmt(Stmt *S) {
    S->dump(llvm::errs(), CGF.getContext());
    llvm_unreachable("Stmt can't have complex result type!");
  }

  mlir::Value VisitExpr(Expr *E) {
    // Crashing here for "ScalarExprClassName"? Please implement
    // VisitScalarExprClassName(...) to get this working.
    emitError(CGF.getLoc(E->getExprLoc()), "scalar exp no implemented: '")
        << E->getStmtClassName() << "'";
    assert(0 && "shouldn't be here!");
    return {};
  }

  mlir::Value VisitConstantExpr(ConstantExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr()); }
  mlir::Value
  VisitSubstnonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCoawaitExpr(CoawaitExpr *S) {
    return CGF.buildCoawaitExpr(*S).getScalarVal();
  }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *S) { llvm_unreachable("NYI"); }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }

  // Leaves.
  mlir::Value VisitIntegerLiteral(const IntegerLiteral *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        Builder.getAttr<mlir::cir::IntAttr>(Ty, E->getValue()));
  }

  mlir::Value VisitFixedPointLiteral(const FixedPointLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitFloatingLiteral(const FloatingLiteral *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        Builder.getFloatAttr(Ty, E->getValue()));
  }
  mlir::Value VisitCharacterLiteral(const CharacterLiteral *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    auto loc = CGF.getLoc(E->getExprLoc());
    auto init = mlir::cir::IntAttr::get(Ty, E->getValue());
    return Builder.create<mlir::cir::ConstantOp>(loc, Ty, init);
  }
  mlir::Value VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    mlir::Type Ty = CGF.getCIRType(E->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty, Builder.getCIRBoolAttr(E->getValue()));
  }

  mlir::Value VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    if (E->getType()->isVoidType())
      return nullptr;

    return buildNullValue(E->getType(), CGF.getLoc(E->getSourceRange()));
  }
  mlir::Value VisitGNUNullExpr(const GNUNullExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitOffsetOfExpr(OffsetOfExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *E);
  mlir::Value VisitAddrLabelExpr(const AddrLabelExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSizeOfPackExpr(SizeOfPackExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitPseudoObjectExpr(PseudoObjectExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSYCLUniqueStableNameExpr(SYCLUniqueStableNameExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    llvm_unreachable("NYI");
  }

  /// Emits the address of the l-value, then loads and returns the result.
  mlir::Value buildLoadOfLValue(const Expr *E) {
    LValue LV = CGF.buildLValue(E);
    // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
    return CGF.buildLoadOfLValue(LV, E->getExprLoc()).getScalarVal();    
  }

  mlir::Value buildLoadOfLValue(LValue LV, SourceLocation Loc) {
    return CGF.buildLoadOfLValue(LV, Loc).getScalarVal();
  }

  // l-values
  mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
    if (CIRGenFunction::ConstantEmission Constant = CGF.tryEmitAsConstant(E)) {
      return CGF.buildScalarConstant(Constant, E);
    }
    return buildLoadOfLValue(E);
  }

  mlir::Value VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCIVarRefExpr(ObjCIvarRefExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCMessageExpr(ObjCMessageExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCIsaExpr(ObjCIsaExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitObjCAvailabilityCheckExpr(ObjCAvailabilityCheckExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    // Do we need anything like TestAndClearIgnoreResultAssign()?
    assert(!E->getBase()->getType()->isVectorType() &&
           "vector types not implemented");

    // Emit subscript expressions in rvalue context's.  For most cases, this
    // just loads the lvalue formed by the subscript expr.  However, we have to
    // be careful, because the base of a vector subscript is occasionally an
    // rvalue, so we can't get it as an lvalue.
    return buildLoadOfLValue(E);
  }

  mlir::Value VisitMatrixSubscriptExpr(MatrixSubscriptExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitConvertVectorExpr(ConvertVectorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitMemberExpr(MemberExpr *E);
  mlir::Value VisitExtVectorelementExpr(Expr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitCompoundLiteralEpxr(CompoundLiteralExpr *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitInitListExpr(InitListExpr *E);

  mlir::Value VisitArrayInitIndexExpr(ArrayInitIndexExpr *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return buildNullValue(E->getType(), CGF.getLoc(E->getSourceRange()));
  }
  mlir::Value VisitExplicitCastExpr(ExplicitCastExpr *E) {
    return VisitCastExpr(E);
  }
  mlir::Value VisitCastExpr(CastExpr *E);
  mlir::Value VisitCallExpr(const CallExpr *E);
  mlir::Value VisitStmtExpr(StmtExpr *E) { llvm_unreachable("NYI"); }

  // Unary Operators.
  mlir::Value VisitUnaryPostDec(const UnaryOperator *E) {
    LValue LV = buildLValue(E->getSubExpr());
    return buildScalarPrePostIncDec(E, LV, false, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *E) {
    LValue LV = buildLValue(E->getSubExpr());
    return buildScalarPrePostIncDec(E, LV, true, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *E) {
    LValue LV = buildLValue(E->getSubExpr());
    return buildScalarPrePostIncDec(E, LV, false, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *E) {
    LValue LV = buildLValue(E->getSubExpr());
    return buildScalarPrePostIncDec(E, LV, true, true);
  }
  mlir::Value buildScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                                       bool isInc, bool isPre) {
    assert(!CGF.getLangOpts().OpenMP && "Not implemented");
    QualType type = E->getSubExpr()->getType();

    int amount = (isInc ? 1 : -1);
    bool atomicPHI = false;
    mlir::Value value{};
    mlir::Value input{};

    if (const AtomicType *atomicTy = type->getAs<AtomicType>()) {
      llvm_unreachable("no atomics inc/dec yet");
    } else {
      value = buildLoadOfLValue(LV, E->getExprLoc());
      input = value;
    }

    // NOTE: When possible, more frequent cases are handled first.

    // Special case of integer increment that we have to check first: bool++.
    // Due to promotion rules, we get:
    //   bool++ -> bool = bool + 1
    //          -> bool = (int)bool + 1
    //          -> bool = ((int)bool + 1 != 0)
    // An interesting aspect of this is that increment is always true.
    // Decrement does not have this property.
    if (isInc && type->isBooleanType()) {
      llvm_unreachable("inc simplification for booleans not implemented yet");

      // NOTE: We likely want the code below, but loading/store booleans need to
      // work first. See CIRGenFunction::buildFromMemory().
      value = Builder.create<mlir::cir::ConstantOp>(
          CGF.getLoc(E->getExprLoc()), CGF.getCIRType(type),
          Builder.getCIRBoolAttr(true));
    } else if (type->isIntegerType()) {
      QualType promotedType;
      bool canPerformLossyDemotionCheck = false;
      if (CGF.getContext().isPromotableIntegerType(type)) {
        promotedType = CGF.getContext().getPromotedIntegerType(type);
        assert(promotedType != type && "Shouldn't promote to the same type.");
        canPerformLossyDemotionCheck = true;
        canPerformLossyDemotionCheck &=
            CGF.getContext().getCanonicalType(type) !=
            CGF.getContext().getCanonicalType(promotedType);
        canPerformLossyDemotionCheck &=
            PromotionIsPotentiallyEligibleForImplicitIntegerConversionCheck(
                type, promotedType);

        // TODO(cir): Currently, we store bitwidths in CIR types only for
        // integers. This might also be required for other types.
        auto srcCirTy = ConvertType(type).dyn_cast<mlir::cir::IntType>();
        auto promotedCirTy = ConvertType(type).dyn_cast<mlir::cir::IntType>();
        assert(srcCirTy && promotedCirTy && "Expected integer type");

        assert(
            (!canPerformLossyDemotionCheck ||
             type->isSignedIntegerOrEnumerationType() ||
             promotedType->isSignedIntegerOrEnumerationType() ||
             srcCirTy.getWidth() == promotedCirTy.getWidth()) &&
            "The following check expects that if we do promotion to different "
            "underlying canonical type, at least one of the types (either "
            "base or promoted) will be signed, or the bitwidths will match.");
      }

      if (CGF.SanOpts.hasOneOf(
              SanitizerKind::ImplicitIntegerArithmeticValueChange) &&
          canPerformLossyDemotionCheck) {
        llvm_unreachable(
            "perform lossy demotion case for inc/dec not implemented yet");
      } else if (E->canOverflow() && type->isSignedIntegerOrEnumerationType()) {
        value = buildIncDecConsiderOverflowBehavior(E, value, isInc);
      } else if (E->canOverflow() && type->isUnsignedIntegerType() &&
                 CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow)) {
        llvm_unreachable(
            "unsigned integer overflow sanitized inc/dec not implemented");
      } else {
        auto Kind = E->isIncrementOp() ? mlir::cir::UnaryOpKind::Inc
                                       : mlir::cir::UnaryOpKind::Dec;
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        value = buildUnaryOp(E, Kind, input);
      }
      // Next most common: pointer increment.
    } else if (const PointerType *ptr = type->getAs<PointerType>()) {
      QualType type = ptr->getPointeeType();
      if (const VariableArrayType *vla =
              CGF.getContext().getAsVariableArrayType(type)) {
        // VLA types don't have constant size.
        llvm_unreachable("NYI");
      } else if (type->isFunctionType()) {
        // Arithmetic on function pointers (!) is just +-1.
        llvm_unreachable("NYI");
      } else {
        // For everything else, we can just do a simple increment.
        auto loc = CGF.getLoc(E->getSourceRange());
        auto &builder = CGF.getBuilder();
        auto amt = builder.getSInt32(amount, loc);
        if (CGF.getLangOpts().isSignedOverflowDefined()) {
          llvm_unreachable("NYI");
        } else {
          value = builder.create<mlir::cir::PtrStrideOp>(loc, value.getType(),
                                                         value, amt);
          assert(!UnimplementedFeature::emitCheckedInBoundsGEP());
        }
      }
    } else if (type->isVectorType()) {
      llvm_unreachable("no vector inc/dec yet");
    } else if (type->isRealFloatingType()) {
      auto isFloatOrDouble = type->isSpecificBuiltinType(BuiltinType::Float) ||
                             type->isSpecificBuiltinType(BuiltinType::Double);
      assert(isFloatOrDouble && "Non-float/double NYI");

      // Create the inc/dec operation.
      auto kind =
          (isInc ? mlir::cir::UnaryOpKind::Inc : mlir::cir::UnaryOpKind::Dec);
      value = buildUnaryOp(E, kind, input);

    } else if (type->isFixedPointType()) {
      llvm_unreachable("no fixed point inc/dec yet");
    } else {
      assert(type->castAs<ObjCObjectPointerType>());
      llvm_unreachable("no objc pointer type inc/dec yet");
    }

    if (atomicPHI) {
      llvm_unreachable("NYI");
    }

    CIRGenFunction::SourceLocRAIIObject sourceloc{
        CGF, CGF.getLoc(E->getSourceRange())};

    // Store the updated result through the lvalue
    if (LV.isBitField())
      llvm_unreachable("no bitfield inc/dec yet");
    else
      CGF.buildStoreThroughLValue(RValue::get(value), LV);

    // If this is a postinc, return the value read from memory, otherwise use
    // the updated value.
    return isPre ? value : input;
  }

  mlir::Value buildIncDecConsiderOverflowBehavior(const UnaryOperator *E,
                                                  mlir::Value InVal,
                                                  bool IsInc) {
    // NOTE(CIR): The SignedOverflowBehavior is attached to the global ModuleOp
    // and the nsw behavior is handled during lowering.
    auto Kind = E->isIncrementOp() ? mlir::cir::UnaryOpKind::Inc
                                   : mlir::cir::UnaryOpKind::Dec;
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      return buildUnaryOp(E, Kind, InVal);
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return buildUnaryOp(E, Kind, InVal);
      llvm_unreachable(
          "inc/dec overflow behavior SOB_Undefined not implemented yet");
      break;
    case LangOptions::SOB_Trapping:
      if (!E->canOverflow())
        return buildUnaryOp(E, Kind, InVal);
      llvm_unreachable(
          "inc/dec overflow behavior SOB_Trapping not implemented yet");
      break;
    }
  }

  mlir::Value VisitUnaryAddrOf(const UnaryOperator *E) {
    assert(!llvm::isa<MemberPointerType>(E->getType()) && "not implemented");
    return CGF.buildLValue(E->getSubExpr()).getPointer();
  }

  mlir::Value VisitUnaryDeref(const UnaryOperator *E) {
    if (E->getType()->isVoidType())
      return Visit(E->getSubExpr()); // the actual value should be unused
    return buildLoadOfLValue(E);
  }
  mlir::Value VisitUnaryPlus(const UnaryOperator *E) {
    // NOTE(cir): QualType function parameter still not used, so don´t replicate
    // it here yet.
    QualType promotionTy = getPromotionType(E->getSubExpr()->getType());
    auto result = VisitPlus(E, promotionTy);
    if (result && !promotionTy.isNull())
      assert(0 && "not implemented yet");
    return buildUnaryOp(E, mlir::cir::UnaryOpKind::Plus, result);
  }

  mlir::Value VisitPlus(const UnaryOperator *E, QualType PromotionType) {
    // This differs from gcc, though, most likely due to a bug in gcc.
    TestAndClearIgnoreResultAssign();
    if (!PromotionType.isNull())
      assert(0 && "scalar promotion not implemented yet");
    return Visit(E->getSubExpr());
  }

  mlir::Value VisitUnaryMinus(const UnaryOperator *E) {
    // NOTE(cir): QualType function parameter still not used, so don´t replicate
    // it here yet.
    QualType promotionTy = getPromotionType(E->getSubExpr()->getType());
    auto result = VisitMinus(E, promotionTy);
    if (result && !promotionTy.isNull())
      assert(0 && "not implemented yet");
    return buildUnaryOp(E, mlir::cir::UnaryOpKind::Minus, result);
  }

  mlir::Value VisitMinus(const UnaryOperator *E, QualType PromotionType) {
    TestAndClearIgnoreResultAssign();
    if (!PromotionType.isNull())
      assert(0 && "scalar promotion not implemented yet");

    // NOTE: LLVM codegen will lower this directly to either a FNeg
    // or a Sub instruction.  In CIR this will be handled later in LowerToLLVM.

    return Visit(E->getSubExpr());
  }

  mlir::Value VisitUnaryNot(const UnaryOperator *E) {
    TestAndClearIgnoreResultAssign();
    mlir::Value op = Visit(E->getSubExpr());
    return buildUnaryOp(E, mlir::cir::UnaryOpKind::Not, op);
  }

  mlir::Value VisitUnaryLNot(const UnaryOperator *E);
  mlir::Value VisitUnaryReal(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryImag(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryExtension(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value buildUnaryOp(const UnaryOperator *E, mlir::cir::UnaryOpKind kind,
                           mlir::Value input) {
    return Builder.create<mlir::cir::UnaryOp>(
        CGF.getLoc(E->getSourceRange().getBegin()),
        CGF.getCIRType(E->getType()), kind, input);
  }

  // C++
  mlir::Value VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSourceLocExpr(SourceLocExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    CIRGenFunction::CXXDefaultArgExprScope Scope(CGF, DAE);
    return Visit(DAE->getExpr());
  }
  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
    CIRGenFunction::CXXDefaultInitExprScope Scope(CGF, DIE);
    return Visit(DIE->getExpr());
  }

  mlir::Value VisitCXXThisExpr(CXXThisExpr *TE) { return CGF.LoadCXXThis(); }

  mlir::Value VisitExprWithCleanups(ExprWithCleanups *E);
  mlir::Value VisitCXXNewExpr(const CXXNewExpr *E) {
    return CGF.buildCXXNewExpr(E);
  }
  mlir::Value VisitCXXDeleteExpr(const CXXDeleteExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitTypeTraitExpr(const TypeTraitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value
  VisitConceptSpecializationExpr(const ConceptSpecializationExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitRequiresExpr(const RequiresExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitExpressionTraitExpr(const ExpressionTraitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *E) {
    return buildNullValue(E->getType(), CGF.getLoc(E->getSourceRange()));
  }
  mlir::Value VisitCXXThrowExpr(CXXThrowExpr *E) {
    CGF.buildCXXThrowExpr(E);
    return nullptr;
  }
  mlir::Value VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
    llvm_unreachable("NYI");
  }

  /// Perform a pointer to boolean conversion.
  mlir::Value buildPointerToBoolConversion(mlir::Value V, QualType QT) {
    // TODO(cir): comparing the ptr to null is done when lowering CIR to LLVM.
    // We might want to have a separate pass for these types of conversions.
    return CGF.getBuilder().createPtrToBoolCast(V);
  }

  // Comparisons.
#define VISITCOMP(CODE)                                                        \
  mlir::Value VisitBin##CODE(const BinaryOperator *E) { return buildCmp(E); }
  VISITCOMP(LT)
  VISITCOMP(GT)
  VISITCOMP(LE)
  VISITCOMP(GE)
  VISITCOMP(EQ)
  VISITCOMP(NE)
#undef VISITCOMP

  mlir::Value VisitBinAssign(const BinaryOperator *E);
  mlir::Value VisitBinLAnd(const BinaryOperator *B);
  mlir::Value VisitBinLOr(const BinaryOperator *B);
  mlir::Value VisitBinComma(const BinaryOperator *E) {
    CGF.buildIgnoredExpr(E->getLHS());
    // NOTE: We don't need to EnsureInsertPoint() like LLVM codegen.
    return Visit(E->getRHS());
  }

  mlir::Value VisitBinPtrMemD(const Expr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitBinPtrMemI(const Expr *E) { llvm_unreachable("NYI"); }

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *E) {
    llvm_unreachable("NYI");
  }

  // Other Operators.
  mlir::Value VisitBlockExpr(const BlockExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *E);
  mlir::Value VisitChooseExpr(ChooseExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitVAArgExpr(VAArgExpr *VE);
  mlir::Value VisitObjCStringLiteral(const ObjCStringLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCBoxedExpr(ObjCBoxedExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitAsTypeExpr(AsTypeExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitAtomicExpr(AtomicExpr *E) { llvm_unreachable("NYI"); }

  // Emit a conversion from the specified type to the specified destination
  // type, both of which are CIR scalar types.
  struct ScalarConversionOpts {
    bool TreatBooleanAsSigned;
    bool EmitImplicitIntegerTruncationChecks;
    bool EmitImplicitIntegerSignChangeChecks;

    ScalarConversionOpts()
        : TreatBooleanAsSigned(false),
          EmitImplicitIntegerTruncationChecks(false),
          EmitImplicitIntegerSignChangeChecks(false) {}

    ScalarConversionOpts(clang::SanitizerSet SanOpts)
        : TreatBooleanAsSigned(false),
          EmitImplicitIntegerTruncationChecks(
              SanOpts.hasOneOf(SanitizerKind::ImplicitIntegerTruncation)),
          EmitImplicitIntegerSignChangeChecks(
              SanOpts.has(SanitizerKind::ImplicitIntegerSignChange)) {}
  };
  mlir::Value buildScalarCast(mlir::Value Src, QualType SrcType,
                              QualType DstType, mlir::Type SrcTy,
                              mlir::Type DstTy, ScalarConversionOpts Opts);

  BinOpInfo buildBinOps(const BinaryOperator *E) {
    BinOpInfo Result;
    Result.LHS = Visit(E->getLHS());
    Result.RHS = Visit(E->getRHS());
    Result.Ty = E->getType();
    Result.Opcode = E->getOpcode();
    Result.Loc = E->getSourceRange();
    // TODO: Result.FPFeatures
    Result.E = E;
    return Result;
  }

  mlir::Value buildMul(const BinOpInfo &Ops);
  mlir::Value buildDiv(const BinOpInfo &Ops);
  mlir::Value buildRem(const BinOpInfo &Ops);
  mlir::Value buildAdd(const BinOpInfo &Ops);
  mlir::Value buildSub(const BinOpInfo &Ops);
  mlir::Value buildShl(const BinOpInfo &Ops);
  mlir::Value buildShr(const BinOpInfo &Ops);
  mlir::Value buildAnd(const BinOpInfo &Ops);
  mlir::Value buildXor(const BinOpInfo &Ops);
  mlir::Value buildOr(const BinOpInfo &Ops);

  LValue buildCompoundAssignLValue(
      const CompoundAssignOperator *E,
      mlir::Value (ScalarExprEmitter::*F)(const BinOpInfo &),
      mlir::Value &Result);
  mlir::Value
  buildCompoundAssign(const CompoundAssignOperator *E,
                      mlir::Value (ScalarExprEmitter::*F)(const BinOpInfo &));

  // TODO(cir): Candidate to be in a common AST helper between CIR and LLVM
  // codegen.
  QualType getPromotionType(QualType Ty) {
    if (auto *CT = Ty->getAs<ComplexType>()) {
      llvm_unreachable("NYI");
    }
    if (Ty.UseExcessPrecision(CGF.getContext()))
      llvm_unreachable("NYI");
    return QualType();
  }

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) {                          \
    return build##OP(buildBinOps(E));                                          \
  }                                                                            \
  mlir::Value VisitBin##OP##Assign(const CompoundAssignOperator *E) {          \
    return buildCompoundAssign(E, &ScalarExprEmitter::build##OP);              \
  }

  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
  HANDLEBINOP(Rem)
  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
  HANDLEBINOP(Shl)
  HANDLEBINOP(Shr)
  HANDLEBINOP(And)
  HANDLEBINOP(Xor)
  HANDLEBINOP(Or)
#undef HANDLEBINOP

  mlir::Value buildCmp(const BinaryOperator *E) {
    mlir::Value Result;
    QualType LHSTy = E->getLHS()->getType();
    QualType RHSTy = E->getRHS()->getType();

    if (const MemberPointerType *MPT = LHSTy->getAs<MemberPointerType>()) {
      assert(0 && "not implemented");
    } else if (!LHSTy->isAnyComplexType() && !RHSTy->isAnyComplexType()) {
      BinOpInfo BOInfo = buildBinOps(E);
      mlir::Value LHS = BOInfo.LHS;
      mlir::Value RHS = BOInfo.RHS;

      if (LHSTy->isVectorType()) {
        // Cannot handle any vector just yet.
        assert(0 && "not implemented");
        // If AltiVec, the comparison results in a numeric type, so we use
        // intrinsics comparing vectors and giving 0 or 1 as a result
        if (!E->getType()->isVectorType())
          assert(0 && "not implemented");
      }
      if (BOInfo.isFixedPointOp()) {
        assert(0 && "not implemented");
      } else {
        // FIXME(cir): handle another if above for CIR equivalent on
        // LHSTy->hasSignedIntegerRepresentation()

        // Unsigned integers and pointers.
        if (CGF.CGM.getCodeGenOpts().StrictVTablePointers &&
            LHS.getType().isa<mlir::cir::PointerType>() &&
            RHS.getType().isa<mlir::cir::PointerType>()) {
          llvm_unreachable("NYI");
        }

        mlir::cir::CmpOpKind Kind;
        switch (E->getOpcode()) {
        case BO_LT:
          Kind = mlir::cir::CmpOpKind::lt;
          break;
        case BO_GT:
          Kind = mlir::cir::CmpOpKind::gt;
          break;
        case BO_LE:
          Kind = mlir::cir::CmpOpKind::le;
          break;
        case BO_GE:
          Kind = mlir::cir::CmpOpKind::ge;
          break;
        case BO_EQ:
          Kind = mlir::cir::CmpOpKind::eq;
          break;
        case BO_NE:
          Kind = mlir::cir::CmpOpKind::ne;
          break;
        default:
          llvm_unreachable("unsupported");
        }

        return Builder.create<mlir::cir::CmpOp>(CGF.getLoc(BOInfo.Loc),
                                                CGF.getCIRType(BOInfo.Ty), Kind,
                                                BOInfo.LHS, BOInfo.RHS);
      }

      // If this is a vector comparison, sign extend the result to the
      // appropriate vector integer type and return it (don't convert to
      // bool).
      if (LHSTy->isVectorType())
        assert(0 && "not implemented");
    } else { // Complex Comparison: can only be an equality comparison.
      assert(0 && "not implemented");
    }

    return buildScalarConversion(Result, CGF.getContext().BoolTy, E->getType(),
                                 E->getExprLoc());
  }

  mlir::Value buildFloatToBoolConversion(mlir::Value src, mlir::Location loc) {
    auto boolTy = Builder.getBoolTy();
    return Builder.create<mlir::cir::CastOp>(
        loc, boolTy, mlir::cir::CastKind::float_to_bool, src);
  }

  mlir::Value buildIntToBoolConversion(mlir::Value srcVal, mlir::Location loc) {
    // Because of the type rules of C, we often end up computing a
    // logical value, then zero extending it to int, then wanting it
    // as a logical value again.
    // TODO: optimize this common case here or leave it for later
    // CIR passes?
    mlir::Type boolTy = CGF.getCIRType(CGF.getContext().BoolTy);
    return Builder.create<mlir::cir::CastOp>(
        loc, boolTy, mlir::cir::CastKind::int_to_bool, srcVal);
  }

  /// Convert the specified expression value to a boolean (!cir.bool) truth
  /// value. This is equivalent to "Val != 0".
  mlir::Value buildConversionToBool(mlir::Value Src, QualType SrcType,
                                    mlir::Location loc) {
    assert(SrcType.isCanonical() && "EmitScalarConversion strips typedefs");

    if (SrcType->isRealFloatingType())
      return buildFloatToBoolConversion(Src, loc);

    if (auto *MPT = llvm::dyn_cast<MemberPointerType>(SrcType))
      assert(0 && "not implemented");

    if (SrcType->isIntegerType())
      return buildIntToBoolConversion(Src, loc);

    assert(Src.getType().isa<::mlir::cir::PointerType>());
    return buildPointerToBoolConversion(Src, SrcType);
  }

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  /// TODO: do we need ScalarConversionOpts here? Should be done in another
  /// pass.
  mlir::Value
  buildScalarConversion(mlir::Value Src, QualType SrcType, QualType DstType,
                        SourceLocation Loc,
                        ScalarConversionOpts Opts = ScalarConversionOpts()) {
    // All conversions involving fixed point types should be handled by the
    // buildFixedPoint family functions. This is done to prevent bloating up
    // this function more, and although fixed point numbers are represented by
    // integers, we do not want to follow any logic that assumes they should be
    // treated as integers.
    // TODO(leonardchan): When necessary, add another if statement checking for
    // conversions to fixed point types from other types.
    if (SrcType->isFixedPointType()) {
      llvm_unreachable("not implemented");
    } else if (DstType->isFixedPointType()) {
      llvm_unreachable("not implemented");
    }

    SrcType = CGF.getContext().getCanonicalType(SrcType);
    DstType = CGF.getContext().getCanonicalType(DstType);
    if (SrcType == DstType)
      return Src;

    if (DstType->isVoidType())
      return nullptr;

    mlir::Type SrcTy = Src.getType();

    // Handle conversions to bool first, they are special: comparisons against
    // 0.
    if (DstType->isBooleanType())
      return buildConversionToBool(Src, SrcType, CGF.getLoc(Loc));

    mlir::Type DstTy = ConvertType(DstType);

    // Cast from half through float if half isn't a native type.
    if (SrcType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      llvm_unreachable("not implemented");
    }

    // TODO(cir): LLVM codegen ignore conversions like int -> uint,
    // is there anything to be done for CIR here?
    if (SrcTy == DstTy) {
      if (Opts.EmitImplicitIntegerSignChangeChecks)
        llvm_unreachable("not implemented");
      return Src;
    }

    assert(!SrcTy.isa<::mlir::cir::PointerType>() &&
           !DstTy.isa<::mlir::cir::PointerType>() &&
           "Internal error: pointer conversions are handled elsewhere");

    // A scalar can be splatted to an extended vector of the same element type
    if (DstType->isExtVectorType() && !SrcType->isVectorType()) {
      // Sema should add casts to make sure that the source expression's type
      // is the same as the vector's element type (sans qualifiers)
      assert(DstType->castAs<ExtVectorType>()->getElementType().getTypePtr() ==
                 SrcType.getTypePtr() &&
             "Splatted expr doesn't match with vector element type?");

      llvm_unreachable("not implemented");
    }

    if (SrcType->isMatrixType() && DstType->isMatrixType())
      llvm_unreachable("NYI: matrix type to matrix type conversion");
    assert(!SrcType->isMatrixType() && !DstType->isMatrixType() &&
           "Internal error: conversion between matrix type and scalar type");

    // TODO(CIR): Support VectorTypes

    // Finally, we have the arithmetic types: real int/float.
    mlir::Value Res = nullptr;
    mlir::Type ResTy = DstTy;

    // An overflowing conversion has undefined behavior if eitehr the source
    // type or the destination type is a floating-point type. However, we
    // consider the range of representable values for all floating-point types
    // to be [-inf,+inf], so no overflow can ever happen when the destination
    // type is a floating-point type.
    if (CGF.SanOpts.has(SanitizerKind::FloatCastOverflow))
      llvm_unreachable("NYI");

    // Cast to half through float if half isn't a native type.
    if (DstType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      llvm_unreachable("NYI");
    }

    Res = buildScalarCast(Src, SrcType, DstType, SrcTy, DstTy, Opts);

    if (DstTy != ResTy) {
      llvm_unreachable("NYI");
    }

    if (Opts.EmitImplicitIntegerTruncationChecks)
      llvm_unreachable("NYI");

    if (Opts.EmitImplicitIntegerSignChangeChecks)
      llvm_unreachable("NYI");

    return Res;
  }
};

} // namespace

/// Emit the computation of the specified expression of scalar type,
/// ignoring the result.
mlir::Value CIRGenFunction::buildScalarExpr(const Expr *E) {
  assert(E && hasScalarEvaluationKind(E->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(E));
}

[[maybe_unused]] static bool MustVisitNullValue(const Expr *E) {
  // If a null pointer expression's type is the C++0x nullptr_t, then
  // it's not necessarily a simple constant and it must be evaluated
  // for its potential side effects.
  return E->getType()->isNullPtrType();
}

/// If \p E is a widened promoted integer, get its base (unpromoted) type.
static std::optional<QualType> getUnwidenedIntegerType(const ASTContext &Ctx,
                                                       const Expr *E) {
  const Expr *Base = E->IgnoreImpCasts();
  if (E == Base)
    return std::nullopt;

  QualType BaseTy = Base->getType();
  if (!Ctx.isPromotableIntegerType(BaseTy) ||
      Ctx.getTypeSize(BaseTy) >= Ctx.getTypeSize(E->getType()))
    return std::nullopt;

  return BaseTy;
}

/// Check if \p E is a widened promoted integer.
[[maybe_unused]] static bool IsWidenedIntegerOp(const ASTContext &Ctx,
                                                const Expr *E) {
  return getUnwidenedIntegerType(Ctx, E).has_value();
}

/// Check if we can skip the overflow check for \p Op.
[[maybe_unused]] static bool CanElideOverflowCheck(const ASTContext &Ctx,
                                                   const BinOpInfo &Op) {
  assert((isa<UnaryOperator>(Op.E) || isa<BinaryOperator>(Op.E)) &&
         "Expected a unary or binary operator");

  // If the binop has constant inputs and we can prove there is no overflow,
  // we can elide the overflow check.
  if (!Op.mayHaveIntegerOverflow())
    return true;

  // If a unary op has a widened operand, the op cannot overflow.
  if (const auto *UO = dyn_cast<UnaryOperator>(Op.E))
    return !UO->canOverflow();

  // We usually don't need overflow checks for binops with widened operands.
  // Multiplication with promoted unsigned operands is a special case.
  const auto *BO = cast<BinaryOperator>(Op.E);
  auto OptionalLHSTy = getUnwidenedIntegerType(Ctx, BO->getLHS());
  if (!OptionalLHSTy)
    return false;

  auto OptionalRHSTy = getUnwidenedIntegerType(Ctx, BO->getRHS());
  if (!OptionalRHSTy)
    return false;

  QualType LHSTy = *OptionalLHSTy;
  QualType RHSTy = *OptionalRHSTy;

  // This is the simple case: binops without unsigned multiplication, and with
  // widened operands. No overflow check is needed here.
  if ((Op.Opcode != BO_Mul && Op.Opcode != BO_MulAssign) ||
      !LHSTy->isUnsignedIntegerType() || !RHSTy->isUnsignedIntegerType())
    return true;

  // For unsigned multiplication the overflow check can be elided if either one
  // of the unpromoted types are less than half the size of the promoted type.
  unsigned PromotedSize = Ctx.getTypeSize(Op.E->getType());
  return (2 * Ctx.getTypeSize(LHSTy)) < PromotedSize ||
         (2 * Ctx.getTypeSize(RHSTy)) < PromotedSize;
}

/// Emit pointer + index arithmetic.
static mlir::Value buildPointerArithmetic(CIRGenFunction &CGF,
                                          const BinOpInfo &op,
                                          bool isSubtraction) {
  // Must have binary (not unary) expr here.  Unary pointer
  // increment/decrement doesn't use this path.
  const BinaryOperator *expr = cast<BinaryOperator>(op.E);

  mlir::Value pointer = op.LHS;
  Expr *pointerOperand = expr->getLHS();
  mlir::Value index = op.RHS;
  Expr *indexOperand = expr->getRHS();

  // In a subtraction, the LHS is always the pointer.
  if (!isSubtraction && !pointer.getType().isa<mlir::cir::PointerType>()) {
    std::swap(pointer, index);
    std::swap(pointerOperand, indexOperand);
  }

  bool isSigned = indexOperand->getType()->isSignedIntegerOrEnumerationType();  

  // Some versions of glibc and gcc use idioms (particularly in their malloc
  // routines) that add a pointer-sized integer (known to be a pointer value)
  // to a null pointer in order to cast the value back to an integer or as
  // part of a pointer alignment algorithm.  This is undefined behavior, but
  // we'd like to be able to compile programs that use it.
  //
  // Normally, we'd generate a GEP with a null-pointer base here in response
  // to that code, but it's also UB to dereference a pointer created that
  // way.  Instead (as an acknowledged hack to tolerate the idiom) we will
  // generate a direct cast of the integer value to a pointer.
  //
  // The idiom (p = nullptr + N) is not met if any of the following are true:
  //
  //   The operation is subtraction.
  //   The index is not pointer-sized.
  //   The pointer type is not byte-sized.
  //
  if (BinaryOperator::isNullPointerArithmeticExtension(
          CGF.getContext(), op.Opcode, expr->getLHS(), expr->getRHS()))
    llvm_unreachable("null pointer arithmetic extension is NYI");

  if (UnimplementedFeature::dataLayoutGetIndexTypeSizeInBits()) {
    // TODO(cir): original codegen zero/sign-extends the index to the same width
    // as the pointer. Since CIR's pointer stride doesn't care about that, it's
    // skiped here.
    llvm_unreachable("target-specific pointer width is NYI");
  }

  // If this is subtraction, negate the index.
  if (isSubtraction)
    index = CGF.getBuilder().createNeg(index);

  if (CGF.SanOpts.has(SanitizerKind::ArrayBounds))
    llvm_unreachable("array bounds sanitizer is NYI");

  const PointerType *pointerType =
      pointerOperand->getType()->getAs<PointerType>();
  if (!pointerType)
    llvm_unreachable("ObjC is NYI");

  QualType elementType = pointerType->getPointeeType();
  if (const VariableArrayType *vla =
          CGF.getContext().getAsVariableArrayType(elementType))
    llvm_unreachable("VLA pointer arithmetic is NYI");

  // Explicitly handle GNU void* and function pointer arithmetic extensions. The
  // GNU void* casts amount to no-ops since our void* type is i8*, but this is
  // future proof.
  if (elementType->isVoidType() || elementType->isFunctionType())
    llvm_unreachable("GNU void* and func ptr arithmetic extensions are NYI");

  mlir::Type elemTy = CGF.convertTypeForMem(elementType);
  if (CGF.getLangOpts().isSignedOverflowDefined())
    llvm_unreachable("ptr arithmetic with signed overflow is NYI");

  return CGF.buildCheckedInBoundsGEP(elemTy, pointer, index, isSigned,
                                     isSubtraction, op.E->getExprLoc());
}

mlir::Value ScalarExprEmitter::buildMul(const BinOpInfo &Ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Mul,
      Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::buildDiv(const BinOpInfo &Ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Div,
      Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::buildRem(const BinOpInfo &Ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Rem,
      Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::buildAdd(const BinOpInfo &Ops) {
  if (Ops.LHS.getType().isa<mlir::cir::PointerType>() ||
      Ops.RHS.getType().isa<mlir::cir::PointerType>())
    return buildPointerArithmetic(CGF, Ops, /*isSubtraction=*/false);

  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Add,
      Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::buildSub(const BinOpInfo &Ops) {
  // The LHS is always a pointer if either side is.
  if (!Ops.LHS.getType().isa<mlir::cir::PointerType>()) {
    if (Ops.Ty->isSignedIntegerOrEnumerationType()) {
      switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
      case LangOptions::SOB_Defined: {
        llvm_unreachable("NYI");
        return Builder.create<mlir::cir::BinOp>(
            CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty),
            mlir::cir::BinOpKind::Sub, Ops.LHS, Ops.RHS);
      }
      case LangOptions::SOB_Undefined:
        if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return Builder.create<mlir::cir::BinOp>(
              CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty),
              mlir::cir::BinOpKind::Sub, Ops.LHS, Ops.RHS);
        [[fallthrough]];
      case LangOptions::SOB_Trapping:
        if (CanElideOverflowCheck(CGF.getContext(), Ops))
          llvm_unreachable("NYI");
        llvm_unreachable("NYI");
      }
    }

    if (Ops.Ty->isConstantMatrixType()) {
      llvm_unreachable("NYI");
    }

    if (Ops.Ty->isUnsignedIntegerType() &&
        CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
        !CanElideOverflowCheck(CGF.getContext(), Ops))
      llvm_unreachable("NYI");

    assert(!UnimplementedFeature::cirVectorType());
    if (Ops.LHS.getType().isa<mlir::FloatType>()) {
      CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, Ops.FPFeatures);
      return Builder.createFSub(Ops.LHS, Ops.RHS);
    }

    if (Ops.isFixedPointOp())
      llvm_unreachable("NYI");

    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Sub,
        Ops.LHS, Ops.RHS);
  }

  // If the RHS is not a pointer, then we have normal pointer
  // arithmetic.
  if (!Ops.RHS.getType().isa<mlir::cir::PointerType>())
    return buildPointerArithmetic(CGF, Ops, /*isSubtraction=*/true);

  // Otherwise, this is a pointer subtraction

  // Do the raw subtraction part.
  //
  // TODO(cir): note for LLVM lowering out of this; when expanding this into
  // LLVM we shall take VLA's, division by element size, etc.
  //
  // See more in `EmitSub` in CGExprScalar.cpp.
  assert(!UnimplementedFeature::llvmLoweringPtrDiffConsidersPointee());
  return Builder.create<mlir::cir::PtrDiffOp>(CGF.getLoc(Ops.Loc),
                                              CGF.PtrDiffTy, Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::buildShl(const BinOpInfo &Ops) {
  // TODO: This misses out on the sanitizer check below.
  if (Ops.isFixedPointOp())
    llvm_unreachable("NYI");

  // CIR accepts shift between different types, meaning nothing special
  // to be done here. OTOH, LLVM requires the LHS and RHS to be the same type:
  // promote or truncate the RHS to the same size as the LHS.

  bool SanitizeSignedBase = CGF.SanOpts.has(SanitizerKind::ShiftBase) &&
                            Ops.Ty->hasSignedIntegerRepresentation() &&
                            !CGF.getLangOpts().isSignedOverflowDefined() &&
                            !CGF.getLangOpts().CPlusPlus20;
  bool SanitizeUnsignedBase =
      CGF.SanOpts.has(SanitizerKind::UnsignedShiftBase) &&
      Ops.Ty->hasUnsignedIntegerRepresentation();
  bool SanitizeBase = SanitizeSignedBase || SanitizeUnsignedBase;
  bool SanitizeExponent = CGF.SanOpts.has(SanitizerKind::ShiftExponent);

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (CGF.getLangOpts().OpenCL)
    llvm_unreachable("NYI");
  else if ((SanitizeBase || SanitizeExponent) &&
           Ops.LHS.getType().isa<mlir::cir::IntType>()) {
    llvm_unreachable("NYI");
  }

  return Builder.create<mlir::cir::ShiftOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), Ops.LHS, Ops.RHS,
      CGF.getBuilder().getUnitAttr());
}

mlir::Value ScalarExprEmitter::buildShr(const BinOpInfo &Ops) {
  // TODO: This misses out on the sanitizer check below.
  if (Ops.isFixedPointOp())
    llvm_unreachable("NYI");

  // CIR accepts shift between different types, meaning nothing special
  // to be done here. OTOH, LLVM requires the LHS and RHS to be the same type:
  // promote or truncate the RHS to the same size as the LHS.

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (CGF.getLangOpts().OpenCL)
    llvm_unreachable("NYI");
  else if (CGF.SanOpts.has(SanitizerKind::ShiftExponent) &&
           Ops.LHS.getType().isa<mlir::cir::IntType>()) {
    llvm_unreachable("NYI");
  }

  // Note that we don't need to distinguish unsigned treatment at this
  // point since it will be handled later by LLVM lowering.
  return Builder.create<mlir::cir::ShiftOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::buildAnd(const BinOpInfo &Ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::And,
      Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::buildXor(const BinOpInfo &Ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Xor,
      Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::buildOr(const BinOpInfo &Ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(Ops.Loc), CGF.getCIRType(Ops.Ty), mlir::cir::BinOpKind::Or,
      Ops.LHS, Ops.RHS);
}

// Emit code for an explicit or implicit cast.  Implicit
// casts have to handle a more broad range of conversions than explicit
// casts, as they handle things like function to ptr-to-function decay
// etc.
mlir::Value ScalarExprEmitter::VisitCastExpr(CastExpr *CE) {
  Expr *E = CE->getSubExpr();
  QualType DestTy = CE->getType();
  CastKind Kind = CE->getCastKind();

  // These cases are generally not written to ignore the result of evaluating
  // their sub-expressions, so we clear this now.
  bool Ignored = TestAndClearIgnoreResultAssign();
  (void)Ignored;

  // Since almost all cast kinds apply to scalars, this switch doesn't have a
  // default case, so the compiler will warn on a missing case. The cases are
  // in the same order as in the CastKind enum.
  switch (Kind) {
  case clang::CK_Dependent:
    llvm_unreachable("dependent cast kind in CIR gen!");
  case clang::CK_BuiltinFnToFnPtr:
    llvm_unreachable("builtin functions are handled elsewhere");

  case CK_LValueBitCast:
    llvm_unreachable("NYI");
  case CK_ObjCObjectLValueCast:
    llvm_unreachable("NYI");
  case CK_LValueToRValueBitCast:
    llvm_unreachable("NYI");
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_BitCast: {
    auto Src = Visit(const_cast<Expr *>(E));
    mlir::Type DstTy = CGF.convertType(DestTy);

    assert(!UnimplementedFeature::cirVectorType());
    assert(!UnimplementedFeature::addressSpace());
    if (CGF.SanOpts.has(SanitizerKind::CFIUnrelatedCast)) {
      llvm_unreachable("NYI");
    }

    if (CGF.CGM.getCodeGenOpts().StrictVTablePointers) {
      llvm_unreachable("NYI");
    }

    // Update heapallocsite metadata when there is an explicit pointer cast.
    assert(!UnimplementedFeature::addHeapAllocSiteMetadata());

    // If Src is a fixed vector and Dst is a scalable vector, and both have the
    // same element type, use the llvm.vector.insert intrinsic to perform the
    // bitcast.
    assert(!UnimplementedFeature::cirVectorType());

    // If Src is a scalable vector and Dst is a fixed vector, and both have the
    // same element type, use the llvm.vector.extract intrinsic to perform the
    // bitcast.
    assert(!UnimplementedFeature::cirVectorType());

    // Perform VLAT <-> VLST bitcast through memory.
    // TODO: since the llvm.experimental.vector.{insert,extract} intrinsics
    //       require the element types of the vectors to be the same, we
    //       need to keep this around for bitcasts between VLAT <-> VLST where
    //       the element types of the vectors are not the same, until we figure
    //       out a better way of doing these casts.
    assert(!UnimplementedFeature::cirVectorType());
    return CGF.getBuilder().createBitcast(CGF.getLoc(E->getSourceRange()), Src,
                                          DstTy);
  }
  case CK_AddressSpaceConversion:
    llvm_unreachable("NYI");
  case CK_AtomicToNonAtomic:
    llvm_unreachable("NYI");
  case CK_NonAtomicToAtomic:
    llvm_unreachable("NYI");
  case CK_UserDefinedConversion:
    return Visit(const_cast<Expr *>(E));
  case CK_NoOp: {
    auto V = Visit(const_cast<Expr *>(E));
    if (V) {
      // CK_NoOp can model a pointer qualification conversion, which can remove
      // an array bound and change the IR type.
      // FIXME: Once pointee types are removed from IR, remove this.
      auto T = CGF.convertType(DestTy);
      if (T != V.getType())
        assert(0 && "NYI");
    }
    return V;
  }
  case CK_BaseToDerived:
    llvm_unreachable("NYI");
  case CK_DerivedToBase: {
    // The EmitPointerWithAlignment path does this fine; just discard
    // the alignment.
    return CGF.buildPointerWithAlignment(CE).getPointer();
  }
  case CK_Dynamic:
    llvm_unreachable("NYI");
  case CK_ArrayToPointerDecay:
    return CGF.buildArrayToPointerDecay(E).getPointer();
  case CK_FunctionToPointerDecay:
    return buildLValue(E).getPointer();

  case CK_NullToPointer: {
    // FIXME: use MustVisitNullValue(E) and evaluate expr.
    // Note that DestTy is used as the MLIR type instead of a custom
    // nullptr type.
    mlir::Type Ty = CGF.getCIRType(DestTy);
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        mlir::cir::ConstPtrAttr::get(Builder.getContext(), Ty, 0));
  }

  case CK_NullToMemberPointer:
    llvm_unreachable("NYI");
  case CK_ReinterpretMemberPointer:
    llvm_unreachable("NYI");
  case CK_BaseToDerivedMemberPointer:
    llvm_unreachable("NYI");
  case CK_DerivedToBaseMemberPointer:
    llvm_unreachable("NYI");
  case CK_ARCProduceObject:
    llvm_unreachable("NYI");
  case CK_ARCConsumeObject:
    llvm_unreachable("NYI");
  case CK_ARCReclaimReturnedObject:
    llvm_unreachable("NYI");
  case CK_ARCExtendBlockObject:
    llvm_unreachable("NYI");
  case CK_CopyAndAutoreleaseBlockObject:
    llvm_unreachable("NYI");
  case CK_FloatingRealToComplex:
    llvm_unreachable("NYI");
  case CK_FloatingComplexCast:
    llvm_unreachable("NYI");
  case CK_IntegralComplexToFloatingComplex:
    llvm_unreachable("NYI");
  case CK_FloatingComplexToIntegralComplex:
    llvm_unreachable("NYI");
  case CK_ConstructorConversion:
    llvm_unreachable("NYI");
  case CK_ToUnion:
    llvm_unreachable("NYI");

  case CK_LValueToRValue:
    assert(CGF.getContext().hasSameUnqualifiedType(E->getType(), DestTy));
    assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
    return Visit(const_cast<Expr *>(E));

  case CK_IntegralToPointer: {
    auto DestCIRTy = ConvertType(DestTy);
    mlir::Value Src = Visit(const_cast<Expr *>(E));

    // Properly resize by casting to an int of the same size as the pointer.
    // Clang's IntegralToPointer includes 'bool' as the source, but in CIR
    // 'bool' is not an integral type.  So check the source type to get the
    // correct CIR conversion.
    auto MiddleTy = CGF.CGM.getDataLayout().getIntPtrType(DestCIRTy);
    auto MiddleVal = Builder.createCast(E->getType()->isBooleanType()
                                            ? mlir::cir::CastKind::bool_to_int
                                            : mlir::cir::CastKind::integral,
                                        Src, MiddleTy);

    if (CGF.CGM.getCodeGenOpts().StrictVTablePointers)
      llvm_unreachable("NYI");

    return Builder.createIntToPtr(MiddleVal, DestCIRTy);
  }
  case CK_PointerToIntegral: {
    assert(!DestTy->isBooleanType() && "bool should use PointerToBool");
    if (CGF.CGM.getCodeGenOpts().StrictVTablePointers)
      llvm_unreachable("NYI");
    return Builder.createPtrToInt(Visit(E), ConvertType(DestTy));
  }
  case CK_ToVoid: {
    CGF.buildIgnoredExpr(E);
    return nullptr;
  }
  case CK_MatrixCast:
    llvm_unreachable("NYI");
  case CK_VectorSplat:
    llvm_unreachable("NYI");
  case CK_FixedPointCast:
    llvm_unreachable("NYI");
  case CK_FixedPointToBoolean:
    llvm_unreachable("NYI");
  case CK_FixedPointToIntegral:
    llvm_unreachable("NYI");
  case CK_IntegralToFixedPoint:
    llvm_unreachable("NYI");

  case CK_IntegralCast: {
    ScalarConversionOpts Opts;
    if (auto *ICE = dyn_cast<ImplicitCastExpr>(CE)) {
      if (!ICE->isPartOfExplicitCast())
        Opts = ScalarConversionOpts(CGF.SanOpts);
    }
    return buildScalarConversion(Visit(E), E->getType(), DestTy,
                                 CE->getExprLoc(), Opts);
  }

  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingCast:
  case CK_FixedPointToFloating:
  case CK_FloatingToFixedPoint: {
    if (Kind == CK_FixedPointToFloating || Kind == CK_FloatingToFixedPoint)
      llvm_unreachable("Fixed point casts are NYI.");
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, CE);
    return buildScalarConversion(Visit(E), E->getType(), DestTy,
                                 CE->getExprLoc());
  }
  case CK_BooleanToSignedIntegral:
    llvm_unreachable("NYI");

  case CK_IntegralToBoolean: {
    return buildIntToBoolConversion(Visit(E), CGF.getLoc(CE->getSourceRange()));
  }

  case CK_PointerToBoolean:
    return buildPointerToBoolConversion(Visit(E), E->getType());
  case CK_FloatingToBoolean:
    return buildFloatToBoolConversion(Visit(E), CGF.getLoc(E->getExprLoc()));
  case CK_MemberPointerToBoolean:
    llvm_unreachable("NYI");
  case CK_FloatingComplexToReal:
    llvm_unreachable("NYI");
  case CK_IntegralComplexToReal:
    llvm_unreachable("NYI");
  case CK_FloatingComplexToBoolean:
    llvm_unreachable("NYI");
  case CK_IntegralComplexToBoolean:
    llvm_unreachable("NYI");
  case CK_ZeroToOCLOpaqueType:
    llvm_unreachable("NYI");
  case CK_IntToOCLSampler:
    llvm_unreachable("NYI");

  default:
    emitError(CGF.getLoc(CE->getExprLoc()), "cast kind not implemented: '")
        << CE->getCastKindName() << "'";
    return nullptr;
  } // end of switch

  llvm_unreachable("unknown scalar cast");
}

mlir::Value ScalarExprEmitter::VisitCallExpr(const CallExpr *E) {
  if (E->getCallReturnType(CGF.getContext())->isReferenceType())
    return buildLoadOfLValue(E);

  auto V = CGF.buildCallExpr(E).getScalarVal();
  assert(!UnimplementedFeature::buildLValueAlignmentAssumption());
  return V;
}

mlir::Value ScalarExprEmitter::VisitMemberExpr(MemberExpr *E) {
  // TODO(cir): Folding all this constants sound like work for MLIR optimizers,
  // keep assertion for now.
  assert(!UnimplementedFeature::tryEmitAsConstant());
  Expr::EvalResult Result;
  if (E->EvaluateAsInt(Result, CGF.getContext(), Expr::SE_AllowSideEffects))
    assert(0 && "NYI");
  return buildLoadOfLValue(E);
}

/// Emit a conversion from the specified type to the specified destination
/// type, both of which are CIR scalar types.
mlir::Value CIRGenFunction::buildScalarConversion(mlir::Value Src,
                                                  QualType SrcTy,
                                                  QualType DstTy,
                                                  SourceLocation Loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(SrcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(DstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this, builder)
      .buildScalarConversion(Src, SrcTy, DstTy, Loc);
}

/// If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the boolean result in Result.
bool CIRGenFunction::ConstantFoldsToSimpleInteger(const Expr *Cond,
                                                  bool &ResultBool,
                                                  bool AllowLabels) {
  llvm::APSInt ResultInt;
  if (!ConstantFoldsToSimpleInteger(Cond, ResultInt, AllowLabels))
    return false;

  ResultBool = ResultInt.getBoolValue();
  return true;
}

mlir::Value ScalarExprEmitter::VisitInitListExpr(InitListExpr *E) {
  bool Ignore = TestAndClearIgnoreResultAssign();
  (void)Ignore;
  assert(Ignore == false && "init list ignored");
  unsigned NumInitElements = E->getNumInits();

  if (E->hadArrayRangeDesignator())
    llvm_unreachable("NYI");

  if (UnimplementedFeature::cirVectorType())
    llvm_unreachable("NYI");

  if (NumInitElements == 0) {
    // C++11 value-initialization for the scalar.
    llvm_unreachable("NYI");
  }

  return Visit(E->getInit(0));
}

mlir::Value ScalarExprEmitter::VisitUnaryLNot(const UnaryOperator *E) {
  // Perform vector logical not on comparison with zero vector.
  if (E->getType()->isVectorType() &&
      E->getType()->castAs<VectorType>()->getVectorKind() ==
          VectorKind::Generic) {
    llvm_unreachable("NYI");
  }

  // Compare operand to zero.
  mlir::Value boolVal = CGF.evaluateExprAsBool(E->getSubExpr());

  // Invert value.
  boolVal = Builder.createNot(boolVal);

  // ZExt result to the expr type.
  auto dstTy = ConvertType(E->getType());
  if (dstTy.isa<mlir::cir::IntType>())
    return Builder.createBoolToInt(boolVal, dstTy);
  if (dstTy.isa<mlir::cir::BoolType>())
    return boolVal;

  llvm_unreachable("destination type for negation unary operator is NYI");
}

// Conversion from bool, integral, or floating-point to integral or
// floating-point.  Conversions involving other types are handled elsewhere.
// Conversion to bool is handled elsewhere because that's a comparison against
// zero, not a simple cast.
mlir::Value ScalarExprEmitter::buildScalarCast(
    mlir::Value Src, QualType SrcType, QualType DstType, mlir::Type SrcTy,
    mlir::Type DstTy, ScalarConversionOpts Opts) {
  assert(!SrcType->isMatrixType() && !DstType->isMatrixType() &&
         "Internal error: matrix types not handled by this function.");
  if (SrcTy.isa<mlir::IntegerType>() || DstTy.isa<mlir::IntegerType>())
    llvm_unreachable("Obsolete code. Don't use mlir::IntegerType with CIR.");

  std::optional<mlir::cir::CastKind> CastKind;

  if (SrcType->isBooleanType()) {
    if (Opts.TreatBooleanAsSigned)
      llvm_unreachable("NYI: signed bool");
    if (CGF.getBuilder().isInt(DstTy)) {
      CastKind = mlir::cir::CastKind::bool_to_int;
    } else if (DstTy.isa<mlir::FloatType>()) {
      CastKind = mlir::cir::CastKind::bool_to_float;
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else if (CGF.getBuilder().isInt(SrcTy)) {
    if (CGF.getBuilder().isInt(DstTy)) {
      CastKind = mlir::cir::CastKind::integral;
    } else if (DstTy.isa<mlir::FloatType>()) {
      CastKind = mlir::cir::CastKind::int_to_float;
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else if (SrcTy.isa<mlir::FloatType>()) {
    if (CGF.getBuilder().isInt(DstTy)) {
      // If we can't recognize overflow as undefined behavior, assume that
      // overflow saturates. This protects against normal optimizations if we
      // are compiling with non-standard FP semantics.
      if (!CGF.CGM.getCodeGenOpts().StrictFloatCastOverflow)
        llvm_unreachable("NYI");
      if (Builder.getIsFPConstrained())
        llvm_unreachable("NYI");
      CastKind = mlir::cir::CastKind::float_to_int;
    } else if (DstTy.isa<mlir::FloatType>()) {
      // TODO: split this to createFPExt/createFPTrunc
      return Builder.createFloatingCast(Src, DstTy);
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else {
    llvm_unreachable("Internal error: Cast from unexpected type");
  }

  assert(CastKind.has_value() && "Internal error: CastKind not set.");
  return Builder.create<mlir::cir::CastOp>(Src.getLoc(), DstTy, *CastKind, Src);
}

LValue
CIRGenFunction::buildCompoundAssignmentLValue(const CompoundAssignOperator *E) {
  ScalarExprEmitter Scalar(*this, builder);
  mlir::Value Result;
  switch (E->getOpcode()) {
#define COMPOUND_OP(Op)                                                        \
  case BO_##Op##Assign:                                                        \
    return Scalar.buildCompoundAssignLValue(E, &ScalarExprEmitter::build##Op,  \
                                            Result)
    COMPOUND_OP(Mul);
    COMPOUND_OP(Div);
    COMPOUND_OP(Rem);
    COMPOUND_OP(Add);
    COMPOUND_OP(Sub);
    COMPOUND_OP(Shl);
    COMPOUND_OP(Shr);
    COMPOUND_OP(And);
    COMPOUND_OP(Xor);
    COMPOUND_OP(Or);
#undef COMPOUND_OP

  case BO_PtrMemD:
  case BO_PtrMemI:
  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Add:
  case BO_Sub:
  case BO_Shl:
  case BO_Shr:
  case BO_LT:
  case BO_GT:
  case BO_LE:
  case BO_GE:
  case BO_EQ:
  case BO_NE:
  case BO_Cmp:
  case BO_And:
  case BO_Xor:
  case BO_Or:
  case BO_LAnd:
  case BO_LOr:
  case BO_Assign:
  case BO_Comma:
    llvm_unreachable("Not valid compound assignment operators");
  }
  llvm_unreachable("Unhandled compound assignment operator");
}

LValue ScalarExprEmitter::buildCompoundAssignLValue(
    const CompoundAssignOperator *E,
    mlir::Value (ScalarExprEmitter::*Func)(const BinOpInfo &),
    mlir::Value &Result) {
  QualType LHSTy = E->getLHS()->getType();
  BinOpInfo OpInfo;

  if (E->getComputationResultType()->isAnyComplexType())
    assert(0 && "not implemented");

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.
  OpInfo.RHS = Visit(E->getRHS());
  OpInfo.Ty = E->getComputationResultType();
  OpInfo.Opcode = E->getOpcode();
  OpInfo.FPFeatures = E->getFPFeaturesInEffect(CGF.getLangOpts());
  OpInfo.E = E;
  OpInfo.Loc = E->getSourceRange();

  // Load/convert the LHS
  LValue LHSLV = CGF.buildLValue(E->getLHS());

  if (const AtomicType *atomicTy = LHSTy->getAs<AtomicType>()) {
    assert(0 && "not implemented");
  }

  OpInfo.LHS = buildLoadOfLValue(LHSLV, E->getExprLoc());

  CIRGenFunction::SourceLocRAIIObject sourceloc{
      CGF, CGF.getLoc(E->getSourceRange())};
  SourceLocation Loc = E->getExprLoc();
  OpInfo.LHS =
      buildScalarConversion(OpInfo.LHS, LHSTy, E->getComputationLHSType(), Loc);

  // Expand the binary operator.
  Result = (this->*Func)(OpInfo);

  // Convert the result back to the LHS type,
  // potentially with Implicit Conversion sanitizer check.
  Result = buildScalarConversion(Result, E->getComputationResultType(), LHSTy,
                                 Loc, ScalarConversionOpts(CGF.SanOpts));

  // Store the result value into the LHS lvalue. Bit-fields are handled
  // specially because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after the
  // assignment...'.
  if (LHSLV.isBitField())
    assert(0 && "not yet implemented");
  else
    CGF.buildStoreThroughLValue(RValue::get(Result), LHSLV);

  assert(!CGF.getLangOpts().OpenMP && "Not implemented");
  return LHSLV;
}

mlir::Value ScalarExprEmitter::buildNullValue(QualType Ty, mlir::Location loc) {
  return CGF.buildFromMemory(CGF.CGM.buildNullConstant(Ty, loc), Ty);
}

mlir::Value ScalarExprEmitter::buildCompoundAssign(
    const CompoundAssignOperator *E,
    mlir::Value (ScalarExprEmitter::*Func)(const BinOpInfo &)) {

  bool Ignore = TestAndClearIgnoreResultAssign();
  mlir::Value RHS;
  LValue LHS = buildCompoundAssignLValue(E, Func, RHS);

  // If the result is clearly ignored, return now.
  if (Ignore)
    return {};

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return RHS;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LHS.isVolatile())
    return RHS;

  // Otherwise, reload the value.
  return buildLoadOfLValue(LHS, E->getExprLoc());
}

mlir::Value ScalarExprEmitter::VisitExprWithCleanups(ExprWithCleanups *E) {
  auto scopeLoc = CGF.getLoc(E->getSourceRange());
  auto &builder = CGF.builder;

  auto scope = builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Type &yieldTy, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                              builder.getInsertionBlock()};
        auto scopeYieldVal = Visit(E->getSubExpr());
        if (scopeYieldVal) {
          builder.create<mlir::cir::YieldOp>(loc, scopeYieldVal);
          yieldTy = scopeYieldVal.getType();
        }
      });

  // Defend against dominance problems caused by jumps out of expression
  // evaluation through the shared cleanup block.
  // TODO(cir): Scope.ForceCleanup({&V});
  return scope.getNumResults() > 0 ? scope->getResult(0) : nullptr;
}

mlir::Value ScalarExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  bool Ignore = TestAndClearIgnoreResultAssign();

  mlir::Value RHS;
  LValue LHS;

  switch (E->getLHS()->getType().getObjCLifetime()) {
  case Qualifiers::OCL_Strong:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_Autoreleasing:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_ExplicitNone:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_Weak:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_None:
    // __block variables need to have the rhs evaluated first, plus this should
    // improve codegen just a little.
    RHS = Visit(E->getRHS());
    LHS = buildCheckedLValue(E->getLHS(), CIRGenFunction::TCK_Store);

    // Store the value into the LHS. Bit-fields are handled specially because
    // the result is altered by the store, i.e., [C99 6.5.16p1]
    // 'An assignment expression has the value of the left operand after the
    // assignment...'.
    if (LHS.isBitField()) {
      CGF.buildStoreThroughBitfieldLValue(RValue::get(RHS), LHS, RHS);
    } else {
      CGF.buildNullabilityCheck(LHS, RHS, E->getExprLoc());
      CIRGenFunction::SourceLocRAIIObject loc{CGF,
                                              CGF.getLoc(E->getSourceRange())};
      CGF.buildStoreThroughLValue(RValue::get(RHS), LHS);
    }
  }

  // If the result is clearly ignored, return now.
  if (Ignore)
    return nullptr;

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return RHS;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LHS.isVolatileQualified())
    return RHS;

  // Otherwise, reload the value.
  return buildLoadOfLValue(LHS, E->getExprLoc());
}

/// Return true if the specified expression is cheap enough and side-effect-free
/// enough to evaluate unconditionally instead of conditionally.  This is used
/// to convert control flow into selects in some cases.
/// TODO(cir): can be shared with LLVM codegen.
static bool isCheapEnoughToEvaluateUnconditionally(const Expr *E,
                                                   CIRGenFunction &CGF) {
  // Anything that is an integer or floating point constant is fine.
  return E->IgnoreParens()->isEvaluatable(CGF.getContext());

  // Even non-volatile automatic variables can't be evaluated unconditionally.
  // Referencing a thread_local may cause non-trivial initialization work to
  // occur. If we're inside a lambda and one of the variables is from the scope
  // outside the lambda, that function may have returned already. Reading its
  // locals is a bad idea. Also, these reads may introduce races there didn't
  // exist in the source-level program.
}

mlir::Value ScalarExprEmitter::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *E) {
  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(E->getSourceRange());
  TestAndClearIgnoreResultAssign();

  // Bind the common expression if necessary.
  CIRGenFunction::OpaqueValueMapping binding(CGF, E);

  Expr *condExpr = E->getCond();
  Expr *lhsExpr = E->getTrueExpr();
  Expr *rhsExpr = E->getFalseExpr();

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm.
  bool CondExprBool;
  if (CGF.ConstantFoldsToSimpleInteger(condExpr, CondExprBool)) {
    Expr *live = lhsExpr, *dead = rhsExpr;
    if (!CondExprBool)
      std::swap(live, dead);

    // If the dead side doesn't have labels we need, just emit the Live part.
    if (!CGF.ContainsLabel(dead)) {
      if (CondExprBool)
        assert(!UnimplementedFeature::incrementProfileCounter());
      auto Result = Visit(live);

      // If the live part is a throw expression, it acts like it has a void
      // type, so evaluating it returns a null Value.  However, a conditional
      // with non-void type must return a non-null Value.
      if (!Result && !E->getType()->isVoidType()) {
        llvm_unreachable("NYI");
      }

      return Result;
    }
  }

  // OpenCL: If the condition is a vector, we can treat this condition like
  // the select function.
  if ((CGF.getLangOpts().OpenCL && condExpr->getType()->isVectorType()) ||
      condExpr->getType()->isExtVectorType()) {
    llvm_unreachable("NYI");
  }

  if (condExpr->getType()->isVectorType() ||
      condExpr->getType()->isSveVLSBuiltinType()) {
    llvm_unreachable("NYI");
  }

  // If this is a really simple expression (like x ? 4 : 5), emit this as a
  // select instead of as control flow.  We can only do this if it is cheap and
  // safe to evaluate the LHS and RHS unconditionally.
  if (isCheapEnoughToEvaluateUnconditionally(lhsExpr, CGF) &&
      isCheapEnoughToEvaluateUnconditionally(rhsExpr, CGF)) {
    bool lhsIsVoid = false;
    auto condV = CGF.evaluateExprAsBool(condExpr);
    assert(!UnimplementedFeature::incrementProfileCounter());

    return builder
        .create<mlir::cir::TernaryOp>(
            loc, condV, /*thenBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto lhs = Visit(lhsExpr);
              if (!lhs) {
                lhs = builder.getNullValue(CGF.VoidTy, loc);
                lhsIsVoid = true;
              }
              builder.create<mlir::cir::YieldOp>(loc, lhs);
            },
            /*elseBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto rhs = Visit(rhsExpr);
              if (lhsIsVoid) {
                assert(!rhs && "lhs and rhs types must match");
                rhs = builder.getNullValue(CGF.VoidTy, loc);
              }
              builder.create<mlir::cir::YieldOp>(loc, rhs);
            })
        .getResult();
  }

  mlir::Value condV = CGF.buildOpOnBoolExpr(condExpr, loc, lhsExpr, rhsExpr);
  CIRGenFunction::ConditionalEvaluation eval(CGF);
  SmallVector<mlir::OpBuilder::InsertPoint, 2> insertPoints{};
  mlir::Type yieldTy{};

  auto patchVoidOrThrowSites = [&]() {
    if (insertPoints.empty())
      return;
    // If both arms are void, so be it.
    if (!yieldTy)
      yieldTy = CGF.VoidTy;

    // Insert required yields.
    for (auto &toInsert : insertPoints) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.restoreInsertionPoint(toInsert);

      // Block does not return: build empty yield.
      if (yieldTy.isa<mlir::cir::VoidType>()) {
        builder.create<mlir::cir::YieldOp>(loc);
      } else { // Block returns: set null yield value.
        mlir::Value op0 = builder.getNullValue(yieldTy, loc);
        builder.create<mlir::cir::YieldOp>(loc, op0);
      }
    }
  };

  return builder
      .create<mlir::cir::TernaryOp>(
          loc, condV, /*trueBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                  b.getInsertionBlock()};
            CGF.currLexScope->setAsTernary();

            assert(!UnimplementedFeature::incrementProfileCounter());
            eval.begin(CGF);
            auto lhs = Visit(lhsExpr);
            eval.end(CGF);

            if (lhs) {
              yieldTy = lhs.getType();
              b.create<mlir::cir::YieldOp>(loc, lhs);
              return;
            }
            // If LHS or RHS is a throw or void expression we need to patch arms
            // as to properly match yield types.
            insertPoints.push_back(b.saveInsertionPoint());
          },
          /*falseBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                  b.getInsertionBlock()};
            CGF.currLexScope->setAsTernary();

            assert(!UnimplementedFeature::incrementProfileCounter());
            eval.begin(CGF);
            auto rhs = Visit(rhsExpr);
            eval.end(CGF);

            if (rhs) {
              yieldTy = rhs.getType();
              b.create<mlir::cir::YieldOp>(loc, rhs);
            } else {
              // If LHS or RHS is a throw or void expression we need to patch
              // arms as to properly match yield types.
              insertPoints.push_back(b.saveInsertionPoint());
            }

            patchVoidOrThrowSites();
          })
      .getResult();
}

mlir::Value CIRGenFunction::buildScalarPrePostIncDec(const UnaryOperator *E,
                                                     LValue LV, bool isInc,
                                                     bool isPre) {
  return ScalarExprEmitter(*this, builder)
      .buildScalarPrePostIncDec(E, LV, isInc, isPre);
}

mlir::Value ScalarExprEmitter::VisitBinLAnd(const clang::BinaryOperator *E) {
  if (E->getType()->isVectorType()) {
    llvm_unreachable("NYI");
  }

  bool InstrumentRegions = CGF.CGM.getCodeGenOpts().hasProfileClangInstr();
  mlir::Type ResTy = ConvertType(E->getType());
  mlir::Location Loc = CGF.getLoc(E->getExprLoc());

  // If we have 0 && RHS, see if we can elide RHS, if so, just return 0.
  // If we have 1 && X, just emit X without inserting the control flow.
  bool LHSCondVal;
  if (CGF.ConstantFoldsToSimpleInteger(E->getLHS(), LHSCondVal)) {
    if (LHSCondVal) { // If we have 1 && X, just emit X.

      mlir::Value RHSCond = CGF.evaluateExprAsBool(E->getRHS());

      if (InstrumentRegions) {
        llvm_unreachable("NYI");
      }
      // ZExt result to int or bool.
      return Builder.createZExtOrBitCast(RHSCond.getLoc(), RHSCond, ResTy);
    }
    // 0 && RHS: If it is safe, just elide the RHS, and return 0/false.
    if (!CGF.ContainsLabel(E->getRHS()))
      return Builder.getBool(false, Loc);
  }

  CIRGenFunction::ConditionalEvaluation eval(CGF);

  mlir::Value LHSCondV = CGF.evaluateExprAsBool(E->getLHS());
  auto ResOp = Builder.create<mlir::cir::TernaryOp>(
      Loc, LHSCondV, /*trueBuilder=*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope LexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        mlir::Value RHSCondV = CGF.evaluateExprAsBool(E->getRHS());
        auto res = B.create<mlir::cir::TernaryOp>(
            Loc, RHSCondV, /*trueBuilder*/
            [&](mlir::OpBuilder &B, mlir::Location Loc) {
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    B.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = B.create<mlir::cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       true));
              B.create<mlir::cir::YieldOp>(Loc, res.getRes());
            },
            /*falseBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location Loc) {
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    b.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<mlir::cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       false));
              b.create<mlir::cir::YieldOp>(Loc, res.getRes());
            });
        B.create<mlir::cir::YieldOp>(Loc, res.getResult());
      },
      /*falseBuilder*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        auto res = B.create<mlir::cir::ConstantOp>(
            Loc, Builder.getBoolTy(),
            Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(), false));
        B.create<mlir::cir::YieldOp>(Loc, res.getRes());
      });
  return Builder.createZExtOrBitCast(ResOp.getLoc(), ResOp.getResult(), ResTy);
}

mlir::Value ScalarExprEmitter::VisitBinLOr(const clang::BinaryOperator *E) {
  if (E->getType()->isVectorType()) {
    llvm_unreachable("NYI");
  }

  bool InstrumentRegions = CGF.CGM.getCodeGenOpts().hasProfileClangInstr();
  mlir::Type ResTy = ConvertType(E->getType());
  mlir::Location Loc = CGF.getLoc(E->getExprLoc());

  // If we have 1 || RHS, see if we can elide RHS, if so, just return 1.
  // If we have 0 || X, just emit X without inserting the control flow.
  bool LHSCondVal;
  if (CGF.ConstantFoldsToSimpleInteger(E->getLHS(), LHSCondVal)) {
    if (!LHSCondVal) { // If we have 0 || X, just emit X.

      mlir::Value RHSCond = CGF.evaluateExprAsBool(E->getRHS());

      if (InstrumentRegions) {
        llvm_unreachable("NYI");
      }
      // ZExt result to int or bool.
      return Builder.createZExtOrBitCast(RHSCond.getLoc(), RHSCond, ResTy);
    }
    // 1 || RHS: If it is safe, just elide the RHS, and return 1/true.
    if (!CGF.ContainsLabel(E->getRHS()))
      return Builder.getBool(true, Loc);
  }

  CIRGenFunction::ConditionalEvaluation eval(CGF);

  mlir::Value LHSCondV = CGF.evaluateExprAsBool(E->getLHS());
  auto ResOp = Builder.create<mlir::cir::TernaryOp>(
      Loc, LHSCondV, /*trueBuilder=*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        auto res = B.create<mlir::cir::ConstantOp>(
            Loc, Builder.getBoolTy(),
            Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(), true));
        B.create<mlir::cir::YieldOp>(Loc, res.getRes());
      },
      /*falseBuilder*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope LexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        mlir::Value RHSCondV = CGF.evaluateExprAsBool(E->getRHS());
        auto res = B.create<mlir::cir::TernaryOp>(
            Loc, RHSCondV, /*trueBuilder*/
            [&](mlir::OpBuilder &B, mlir::Location Loc) {
              SmallVector<mlir::Location, 2> Locs;
              if (Loc.isa<mlir::FileLineColLoc>()) {
                Locs.push_back(Loc);
                Locs.push_back(Loc);
              } else if (Loc.isa<mlir::FusedLoc>()) {
                auto fusedLoc = Loc.cast<mlir::FusedLoc>();
                Locs.push_back(fusedLoc.getLocations()[0]);
                Locs.push_back(fusedLoc.getLocations()[1]);
              }
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    B.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = B.create<mlir::cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       true));
              B.create<mlir::cir::YieldOp>(Loc, res.getRes());
            },
            /*falseBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location Loc) {
              SmallVector<mlir::Location, 2> Locs;
              if (Loc.isa<mlir::FileLineColLoc>()) {
                Locs.push_back(Loc);
                Locs.push_back(Loc);
              } else if (Loc.isa<mlir::FusedLoc>()) {
                auto fusedLoc = Loc.cast<mlir::FusedLoc>();
                Locs.push_back(fusedLoc.getLocations()[0]);
                Locs.push_back(fusedLoc.getLocations()[1]);
              }
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    B.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<mlir::cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       false));
              b.create<mlir::cir::YieldOp>(Loc, res.getRes());
            });
        B.create<mlir::cir::YieldOp>(Loc, res.getResult());
      });

  return Builder.createZExtOrBitCast(ResOp.getLoc(), ResOp.getResult(), ResTy);
}

mlir::Value ScalarExprEmitter::VisitVAArgExpr(VAArgExpr *VE) {
  QualType Ty = VE->getType();

  if (Ty->isVariablyModifiedType())
    assert(!UnimplementedFeature::variablyModifiedTypeEmission() && "NYI");

  Address ArgValue = Address::invalid();
  mlir::Value Val = CGF.buildVAArg(VE, ArgValue);

  return Val;
}

/// Return the size or alignment of the type of argument of the sizeof
/// expression as an integer.
mlir::Value ScalarExprEmitter::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *E) {
  QualType TypeToSize = E->getTypeOfArgument();
  if (E->getKind() == UETT_SizeOf) {
    if (const VariableArrayType *VAT =
            CGF.getContext().getAsVariableArrayType(TypeToSize)) {
      llvm_unreachable("NYI");
    }
  } else if (E->getKind() == UETT_OpenMPRequiredSimdAlign) {
    llvm_unreachable("NYI");
  }

  // If this isn't sizeof(vla), the result must be constant; use the constant
  // folding logic so we don't have to duplicate it here.
  return Builder.getConstInt(CGF.getLoc(E->getSourceRange()),
                             E->EvaluateKnownConstInt(CGF.getContext()));
}

mlir::Value CIRGenFunction::buildCheckedInBoundsGEP(
    mlir::Type ElemTy, mlir::Value Ptr, ArrayRef<mlir::Value> IdxList,
    bool SignedIndices, bool IsSubtraction, SourceLocation Loc) {
  mlir::Type PtrTy = Ptr.getType();
  assert(IdxList.size() == 1 && "multi-index ptr arithmetic NYI");
  mlir::Value GEPVal = builder.create<mlir::cir::PtrStrideOp>(
      CGM.getLoc(Loc), PtrTy, Ptr, IdxList[0]);

  // If the pointer overflow sanitizer isn't enabled, do nothing.
  if (!SanOpts.has(SanitizerKind::PointerOverflow))
    return GEPVal;

  // TODO(cir): the unreachable code below hides a substantial amount of code
  // from the original codegen related with pointer overflow sanitizer.
  assert(UnimplementedFeature::pointerOverflowSanitizer());
  llvm_unreachable("pointer overflow sanitizer NYI");
}
