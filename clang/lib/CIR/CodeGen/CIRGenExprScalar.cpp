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

#include "Address.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

struct BinOpInfo {
  mlir::Value LHS;
  mlir::Value RHS;
  SourceRange Loc;
  QualType FullType;             // Type of operands and result
  QualType CompType;             // Type used for computations. Element type
                                 // for vectors, otherwise same as FullType.
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
    auto LHSCI = dyn_cast<cir::ConstantOp>(LHS.getDefiningOp());
    auto RHSCI = dyn_cast<cir::ConstantOp>(RHS.getDefiningOp());
    if (!LHSCI || !RHSCI)
      return true;

    llvm::APInt Result;
    assert(!cir::MissingFeatures::mayHaveIntegerOverflow());
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

  mlir::Type convertType(QualType T) { return CGF.convertType(T); }
  LValue emitLValue(const Expr *E) { return CGF.emitLValue(E); }
  LValue emitCheckedLValue(const Expr *E, CIRGenFunction::TypeCheckKind TCK) {
    return CGF.emitCheckedLValue(E, TCK);
  }

  mlir::Value emitComplexToScalarConversion(mlir::Location Loc, mlir::Value V,
                                            CastKind Kind, QualType DestTy);

  /// Emit a value that corresponds to null for the given type.
  mlir::Value emitNullValue(QualType Ty, mlir::Location loc);

  mlir::Value emitPromotedValue(mlir::Value result, QualType PromotionType) {
    return Builder.createFloatingCast(result, convertType(PromotionType));
  }

  mlir::Value emitUnPromotedValue(mlir::Value result, QualType ExprType) {
    return Builder.createFloatingCast(result, convertType(ExprType));
  }

  mlir::Value emitPromoted(const Expr *E, QualType PromotionType);

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
    llvm_unreachable("NYI");
    return {};
  }

  mlir::Value VisitConstantExpr(ConstantExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr()); }
  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *E) {
    return Visit(E->getReplacement());
  }
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCoawaitExpr(CoawaitExpr *S) {
    return CGF.emitCoawaitExpr(*S).getScalarVal();
  }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *S) {
    return CGF.emitCoyieldExpr(*S).getScalarVal();
  }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *E) {
    llvm_unreachable("NYI");
  }

  // Leaves.
  mlir::Value VisitIntegerLiteral(const IntegerLiteral *E) {
    mlir::Type Ty = CGF.convertType(E->getType());
    return Builder.create<cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        Builder.getAttr<cir::IntAttr>(Ty, E->getValue()));
  }

  mlir::Value VisitFixedPointLiteral(const FixedPointLiteral *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitFloatingLiteral(const FloatingLiteral *E) {
    mlir::Type Ty = CGF.convertType(E->getType());
    assert(mlir::isa<cir::CIRFPTypeInterface>(Ty) &&
           "expect floating-point type");
    return Builder.create<cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty,
        Builder.getAttr<cir::FPAttr>(Ty, E->getValue()));
  }
  mlir::Value VisitCharacterLiteral(const CharacterLiteral *E) {
    mlir::Type Ty = CGF.convertType(E->getType());
    auto loc = CGF.getLoc(E->getExprLoc());
    auto init = cir::IntAttr::get(Ty, E->getValue());
    return Builder.create<cir::ConstantOp>(loc, Ty, init);
  }
  mlir::Value VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    mlir::Type Ty = CGF.convertType(E->getType());
    return Builder.create<cir::ConstantOp>(
        CGF.getLoc(E->getExprLoc()), Ty, Builder.getCIRBoolAttr(E->getValue()));
  }

  mlir::Value VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    if (E->getType()->isVoidType())
      return nullptr;

    return emitNullValue(E->getType(), CGF.getLoc(E->getSourceRange()));
  }
  mlir::Value VisitGNUNullExpr(const GNUNullExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitOffsetOfExpr(OffsetOfExpr *E) {
    // Try folding the offsetof to a constant.
    Expr::EvalResult EVResult;
    if (E->EvaluateAsInt(EVResult, CGF.getContext())) {
      llvm::APSInt Value = EVResult.Val.getInt();
      return Builder.getConstInt(CGF.getLoc(E->getExprLoc()), Value);
    }

    llvm_unreachable("NYI");
  }

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
    if (E->isGLValue())
      llvm_unreachable("NYI");

    // Otherwise, assume the mapping is the scalar directly.
    return CGF.getOrCreateOpaqueRValueMapping(E).getScalarVal();
  }

  /// Emits the address of the l-value, then loads and returns the result.
  mlir::Value emitLoadOfLValue(const Expr *E) {
    LValue LV = CGF.emitLValue(E);
    // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
    return CGF.emitLoadOfLValue(LV, E->getExprLoc()).getScalarVal();
  }

  mlir::Value emitLoadOfLValue(LValue LV, SourceLocation Loc) {
    return CGF.emitLoadOfLValue(LV, Loc).getScalarVal();
  }

  // l-values
  mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
    if (CIRGenFunction::ConstantEmission Constant = CGF.tryEmitAsConstant(E)) {
      return CGF.emitScalarConstant(Constant, E);
    }
    return emitLoadOfLValue(E);
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

    if (E->getBase()->getType()->isVectorType()) {
      assert(!cir::MissingFeatures::scalableVectors() &&
             "NYI: index into scalable vector");
      // Subscript of vector type.  This is handled differently, with a custom
      // operation.
      mlir::Value VecValue = Visit(E->getBase());
      mlir::Value IndexValue = Visit(E->getIdx());
      return CGF.builder.create<cir::VecExtractOp>(
          CGF.getLoc(E->getSourceRange()), VecValue, IndexValue);
    }

    // Just load the lvalue formed by the subscript expression.
    return emitLoadOfLValue(E);
  }

  mlir::Value VisitMatrixSubscriptExpr(MatrixSubscriptExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
    if (E->getNumSubExprs() == 2) {
      // The undocumented form of __builtin_shufflevector.
      mlir::Value InputVec = Visit(E->getExpr(0));
      mlir::Value IndexVec = Visit(E->getExpr(1));
      return CGF.builder.create<cir::VecShuffleDynamicOp>(
          CGF.getLoc(E->getSourceRange()), InputVec, IndexVec);
    } else {
      // The documented form of __builtin_shufflevector, where the indices are
      // a variable number of integer constants. The constants will be stored
      // in an ArrayAttr.
      mlir::Value Vec1 = Visit(E->getExpr(0));
      mlir::Value Vec2 = Visit(E->getExpr(1));
      SmallVector<mlir::Attribute, 8> Indices;
      for (unsigned i = 2; i < E->getNumSubExprs(); ++i) {
        Indices.push_back(
            cir::IntAttr::get(CGF.builder.getSInt64Ty(),
                              E->getExpr(i)
                                  ->EvaluateKnownConstInt(CGF.getContext())
                                  .getSExtValue()));
      }
      return CGF.builder.create<cir::VecShuffleOp>(
          CGF.getLoc(E->getSourceRange()), CGF.convertType(E->getType()), Vec1,
          Vec2, CGF.builder.getArrayAttr(Indices));
    }
  }
  mlir::Value VisitConvertVectorExpr(ConvertVectorExpr *E) {
    // __builtin_convertvector is an element-wise cast, and is implemented as a
    // regular cast. The back end handles casts of vectors correctly.
    return emitScalarConversion(Visit(E->getSrcExpr()),
                                E->getSrcExpr()->getType(), E->getType(),
                                E->getSourceRange().getBegin());
  }

  mlir::Value VisitExtVectorElementExpr(Expr *E) { return emitLoadOfLValue(E); }

  mlir::Value VisitMemberExpr(MemberExpr *E);
  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return emitLoadOfLValue(E);
  }

  mlir::Value VisitInitListExpr(InitListExpr *E);

  mlir::Value VisitArrayInitIndexExpr(ArrayInitIndexExpr *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return emitNullValue(E->getType(), CGF.getLoc(E->getSourceRange()));
  }
  mlir::Value VisitExplicitCastExpr(ExplicitCastExpr *E) {
    return VisitCastExpr(E);
  }
  mlir::Value VisitCastExpr(CastExpr *E);
  mlir::Value VisitCallExpr(const CallExpr *E);

  mlir::Value VisitStmtExpr(StmtExpr *E) {
    assert(!cir::MissingFeatures::stmtExprEvaluation() && "NYI");
    Address retAlloca =
        CGF.emitCompoundStmt(*E->getSubStmt(), !E->getType()->isVoidType());
    if (!retAlloca.isValid())
      return {};

    // FIXME(cir): This is a work around the ScopeOp builder. If we build the
    // ScopeOp before its body, we would be able to create the retAlloca
    // direclty in the parent scope removing the need to hoist it.
    assert(retAlloca.getDefiningOp() && "expected a alloca op");
    CGF.getBuilder().hoistAllocaToParentRegion(
        cast<cir::AllocaOp>(retAlloca.getDefiningOp()));

    return CGF.emitLoadOfScalar(CGF.makeAddrLValue(retAlloca, E->getType()),
                                E->getExprLoc());
  }

  // Unary Operators.
  mlir::Value VisitUnaryPostDec(const UnaryOperator *E) {
    LValue LV = emitLValue(E->getSubExpr());
    return emitScalarPrePostIncDec(E, LV, false, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *E) {
    LValue LV = emitLValue(E->getSubExpr());
    return emitScalarPrePostIncDec(E, LV, true, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *E) {
    LValue LV = emitLValue(E->getSubExpr());
    return emitScalarPrePostIncDec(E, LV, false, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *E) {
    LValue LV = emitLValue(E->getSubExpr());
    return emitScalarPrePostIncDec(E, LV, true, true);
  }
  mlir::Value emitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
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
      value = emitLoadOfLValue(LV, E->getExprLoc());
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
      value = Builder.create<cir::ConstantOp>(CGF.getLoc(E->getExprLoc()),
                                              CGF.convertType(type),
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
        auto srcCirTy = mlir::dyn_cast<cir::IntType>(convertType(type));
        auto promotedCirTy = mlir::dyn_cast<cir::IntType>(convertType(type));
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
        value = emitIncDecConsiderOverflowBehavior(E, value, isInc);
      } else if (E->canOverflow() && type->isUnsignedIntegerType() &&
                 CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow)) {
        llvm_unreachable(
            "unsigned integer overflow sanitized inc/dec not implemented");
      } else {
        auto Kind =
            E->isIncrementOp() ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec;
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        value = emitUnaryOp(E, Kind, input);
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
          value = builder.create<cir::PtrStrideOp>(loc, value.getType(), value,
                                                   amt);
        } else {
          value = builder.create<cir::PtrStrideOp>(loc, value.getType(), value,
                                                   amt);
          assert(!cir::MissingFeatures::emitCheckedInBoundsGEP());
        }
      }
    } else if (type->isVectorType()) {
      llvm_unreachable("no vector inc/dec yet");
    } else if (type->isRealFloatingType()) {
      // TODO(cir): CGFPOptionsRAII
      assert(!cir::MissingFeatures::CGFPOptionsRAII());

      if (type->isHalfType() &&
          !CGF.getContext().getLangOpts().NativeHalfType) {
        // Another special case: half FP increment should be done via float
        if (CGF.getContext().getTargetInfo().useFP16ConversionIntrinsics()) {
          llvm_unreachable("cast via llvm.convert.from.fp16 is NYI");
        } else {
          value = Builder.createCast(CGF.getLoc(E->getExprLoc()),
                                     cir::CastKind::floating, input,
                                     CGF.CGM.FloatTy);
        }
      }

      if (mlir::isa<cir::SingleType, cir::DoubleType>(value.getType())) {
        // Create the inc/dec operation.
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        auto kind = (isInc ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec);
        value = emitUnaryOp(E, kind, value);
      } else {
        // Remaining types are Half, Bfloat16, LongDouble, __ibm128 or
        // __float128. Convert from float.

        llvm::APFloat F(static_cast<float>(amount));
        bool ignored;
        const llvm::fltSemantics *FS;
        // Don't use getFloatTypeSemantics because Half isn't
        // necessarily represented using the "half" LLVM type.
        if (mlir::isa<cir::LongDoubleType>(value.getType()))
          FS = &CGF.getTarget().getLongDoubleFormat();
        else if (mlir::isa<cir::FP16Type>(value.getType()))
          FS = &CGF.getTarget().getHalfFormat();
        else if (mlir::isa<cir::BF16Type>(value.getType()))
          FS = &CGF.getTarget().getBFloat16Format();
        else
          llvm_unreachable("fp128 / ppc_fp128 NYI");
        F.convert(*FS, llvm::APFloat::rmTowardZero, &ignored);

        auto loc = CGF.getLoc(E->getExprLoc());
        auto amt =
            Builder.getConstant(loc, cir::FPAttr::get(value.getType(), F));
        value = Builder.createBinop(value, cir::BinOpKind::Add, amt);
      }

      if (type->isHalfType() &&
          !CGF.getContext().getLangOpts().NativeHalfType) {
        if (CGF.getContext().getTargetInfo().useFP16ConversionIntrinsics()) {
          llvm_unreachable("cast via llvm.convert.to.fp16 is NYI");
        } else {
          value = Builder.createCast(CGF.getLoc(E->getExprLoc()),
                                     cir::CastKind::floating, value,
                                     input.getType());
        }
      }

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
      CGF.emitStoreThroughBitfieldLValue(RValue::get(value), LV, value);
    else
      CGF.emitStoreThroughLValue(RValue::get(value), LV);

    // If this is a postinc, return the value read from memory, otherwise use
    // the updated value.
    return isPre ? value : input;
  }

  mlir::Value emitIncDecConsiderOverflowBehavior(const UnaryOperator *E,
                                                 mlir::Value InVal,
                                                 bool IsInc) {
    // NOTE(CIR): The SignedOverflowBehavior is attached to the global ModuleOp
    // and the nsw behavior is handled during lowering.
    auto Kind =
        E->isIncrementOp() ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec;
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      return emitUnaryOp(E, Kind, InVal);
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return emitUnaryOp(E, Kind, InVal);
      llvm_unreachable(
          "inc/dec overflow behavior SOB_Undefined not implemented yet");
      break;
    case LangOptions::SOB_Trapping:
      if (!E->canOverflow())
        return emitUnaryOp(E, Kind, InVal);
      llvm_unreachable(
          "inc/dec overflow behavior SOB_Trapping not implemented yet");
      break;
    }
  }

  mlir::Value VisitUnaryAddrOf(const UnaryOperator *E) {
    if (llvm::isa<MemberPointerType>(E->getType()))
      return CGF.CGM.emitMemberPointerConstant(E);

    return CGF.emitLValue(E->getSubExpr()).getPointer();
  }

  mlir::Value VisitUnaryDeref(const UnaryOperator *E) {
    if (E->getType()->isVoidType())
      return Visit(E->getSubExpr()); // the actual value should be unused
    return emitLoadOfLValue(E);
  }
  mlir::Value VisitUnaryPlus(const UnaryOperator *E,
                             QualType PromotionType = QualType()) {
    QualType promotionTy = PromotionType.isNull()
                               ? getPromotionType(E->getSubExpr()->getType())
                               : PromotionType;
    auto result = VisitPlus(E, promotionTy);
    if (result && !promotionTy.isNull())
      return emitUnPromotedValue(result, E->getType());
    return result;
  }

  mlir::Value VisitPlus(const UnaryOperator *E,
                        QualType PromotionType = QualType()) {
    // This differs from gcc, though, most likely due to a bug in gcc.
    TestAndClearIgnoreResultAssign();

    mlir::Value operand;
    if (!PromotionType.isNull())
      operand = CGF.emitPromotedScalarExpr(E->getSubExpr(), PromotionType);
    else
      operand = Visit(E->getSubExpr());

    return emitUnaryOp(E, cir::UnaryOpKind::Plus, operand);
  }

  mlir::Value VisitUnaryMinus(const UnaryOperator *E,
                              QualType PromotionType = QualType()) {
    QualType promotionTy = PromotionType.isNull()
                               ? getPromotionType(E->getSubExpr()->getType())
                               : PromotionType;
    auto result = VisitMinus(E, promotionTy);
    if (result && !promotionTy.isNull())
      return emitUnPromotedValue(result, E->getType());
    return result;
  }

  mlir::Value VisitMinus(const UnaryOperator *E, QualType PromotionType) {
    TestAndClearIgnoreResultAssign();

    mlir::Value operand;
    if (!PromotionType.isNull())
      operand = CGF.emitPromotedScalarExpr(E->getSubExpr(), PromotionType);
    else
      operand = Visit(E->getSubExpr());

    // NOTE: LLVM codegen will lower this directly to either a FNeg
    // or a Sub instruction.  In CIR this will be handled later in LowerToLLVM.
    return emitUnaryOp(E, cir::UnaryOpKind::Minus, operand);
  }

  mlir::Value VisitUnaryNot(const UnaryOperator *E) {
    TestAndClearIgnoreResultAssign();
    mlir::Value op = Visit(E->getSubExpr());
    return emitUnaryOp(E, cir::UnaryOpKind::Not, op);
  }

  mlir::Value VisitUnaryLNot(const UnaryOperator *E);
  mlir::Value VisitUnaryReal(const UnaryOperator *E) { return VisitReal(E); }
  mlir::Value VisitUnaryImag(const UnaryOperator *E) { return VisitImag(E); }

  mlir::Value VisitReal(const UnaryOperator *E);
  mlir::Value VisitImag(const UnaryOperator *E);

  mlir::Value VisitUnaryExtension(const UnaryOperator *E) {
    // __extension__ doesn't requred any codegen
    // just forward the value
    return Visit(E->getSubExpr());
  }

  mlir::Value emitUnaryOp(const UnaryOperator *E, cir::UnaryOpKind kind,
                          mlir::Value input) {
    return Builder.create<cir::UnaryOp>(
        CGF.getLoc(E->getSourceRange().getBegin()), input.getType(), kind,
        input);
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
    return CGF.emitCXXNewExpr(E);
  }
  mlir::Value VisitCXXDeleteExpr(const CXXDeleteExpr *E) {
    CGF.emitCXXDeleteExpr(E);
    return {};
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
    return emitNullValue(E->getType(), CGF.getLoc(E->getSourceRange()));
  }
  mlir::Value VisitCXXThrowExpr(CXXThrowExpr *E) {
    CGF.emitCXXThrowExpr(E);
    return nullptr;
  }
  mlir::Value VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
    llvm_unreachable("NYI");
  }

  /// Perform a pointer to boolean conversion.
  mlir::Value emitPointerToBoolConversion(mlir::Value V, QualType QT) {
    // TODO(cir): comparing the ptr to null is done when lowering CIR to LLVM.
    // We might want to have a separate pass for these types of conversions.
    return CGF.getBuilder().createPtrToBoolCast(V);
  }

  // Comparisons.
#define VISITCOMP(CODE)                                                        \
  mlir::Value VisitBin##CODE(const BinaryOperator *E) { return emitCmp(E); }
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
    CGF.emitIgnoredExpr(E->getLHS());
    // NOTE: We don't need to EnsureInsertPoint() like LLVM codegen.
    return Visit(E->getRHS());
  }

  mlir::Value VisitBinPtrMemD(const BinaryOperator *E) {
    return emitLoadOfLValue(E);
  }

  mlir::Value VisitBinPtrMemI(const BinaryOperator *E) {
    return emitLoadOfLValue(E);
  }

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *E) {
    return Visit(E->getSemanticForm());
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
  mlir::Value VisitAtomicExpr(AtomicExpr *E) {
    return CGF.emitAtomicExpr(E).getScalarVal();
  }

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
  mlir::Value emitScalarCast(mlir::Value Src, QualType SrcType,
                             QualType DstType, mlir::Type SrcTy,
                             mlir::Type DstTy, ScalarConversionOpts Opts);

  BinOpInfo emitBinOps(const BinaryOperator *E,
                       QualType PromotionType = QualType()) {
    BinOpInfo Result;
    Result.LHS = CGF.emitPromotedScalarExpr(E->getLHS(), PromotionType);
    Result.RHS = CGF.emitPromotedScalarExpr(E->getRHS(), PromotionType);
    if (!PromotionType.isNull())
      Result.FullType = PromotionType;
    else
      Result.FullType = E->getType();
    Result.CompType = Result.FullType;
    if (const auto *VecType = dyn_cast_or_null<VectorType>(Result.FullType)) {
      Result.CompType = VecType->getElementType();
    }
    Result.Opcode = E->getOpcode();
    Result.Loc = E->getSourceRange();
    // TODO: Result.FPFeatures
    assert(!cir::MissingFeatures::getFPFeaturesInEffect());
    Result.E = E;
    return Result;
  }

  mlir::Value emitMul(const BinOpInfo &Ops);
  mlir::Value emitDiv(const BinOpInfo &Ops);
  mlir::Value emitRem(const BinOpInfo &Ops);
  mlir::Value emitAdd(const BinOpInfo &Ops);
  mlir::Value emitSub(const BinOpInfo &Ops);
  mlir::Value emitShl(const BinOpInfo &Ops);
  mlir::Value emitShr(const BinOpInfo &Ops);
  mlir::Value emitAnd(const BinOpInfo &Ops);
  mlir::Value emitXor(const BinOpInfo &Ops);
  mlir::Value emitOr(const BinOpInfo &Ops);

  LValue emitCompoundAssignLValue(
      const CompoundAssignOperator *E,
      mlir::Value (ScalarExprEmitter::*F)(const BinOpInfo &),
      mlir::Value &Result);
  mlir::Value
  emitCompoundAssign(const CompoundAssignOperator *E,
                     mlir::Value (ScalarExprEmitter::*F)(const BinOpInfo &));

  // TODO(cir): Candidate to be in a common AST helper between CIR and LLVM
  // codegen.
  QualType getPromotionType(QualType Ty) {
    if (auto *CT = Ty->getAs<ComplexType>()) {
      llvm_unreachable("NYI");
    }
    if (Ty.UseExcessPrecision(CGF.getContext())) {
      if (auto *VT = Ty->getAs<VectorType>())
        llvm_unreachable("NYI");
      return CGF.getContext().FloatTy;
    }
    return QualType();
  }

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) {                          \
    QualType promotionTy = getPromotionType(E->getType());                     \
    auto result = emit##OP(emitBinOps(E, promotionTy));                        \
    if (result && !promotionTy.isNull())                                       \
      result = emitUnPromotedValue(result, E->getType());                      \
    return result;                                                             \
  }                                                                            \
  mlir::Value VisitBin##OP##Assign(const CompoundAssignOperator *E) {          \
    return emitCompoundAssign(E, &ScalarExprEmitter::emit##OP);                \
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

  mlir::Value emitCmp(const BinaryOperator *E) {
    mlir::Value Result;
    QualType LHSTy = E->getLHS()->getType();
    QualType RHSTy = E->getRHS()->getType();

    auto ClangCmpToCIRCmp = [](auto ClangCmp) -> cir::CmpOpKind {
      switch (ClangCmp) {
      case BO_LT:
        return cir::CmpOpKind::lt;
      case BO_GT:
        return cir::CmpOpKind::gt;
      case BO_LE:
        return cir::CmpOpKind::le;
      case BO_GE:
        return cir::CmpOpKind::ge;
      case BO_EQ:
        return cir::CmpOpKind::eq;
      case BO_NE:
        return cir::CmpOpKind::ne;
      default:
        llvm_unreachable("unsupported comparison kind");
        return cir::CmpOpKind(-1);
      }
    };

    if (const MemberPointerType *MPT = LHSTy->getAs<MemberPointerType>()) {
      assert(E->getOpcode() == BO_EQ || E->getOpcode() == BO_NE);
      mlir::Value lhs = CGF.emitScalarExpr(E->getLHS());
      mlir::Value rhs = CGF.emitScalarExpr(E->getRHS());
      cir::CmpOpKind kind = ClangCmpToCIRCmp(E->getOpcode());
      Result =
          Builder.createCompare(CGF.getLoc(E->getExprLoc()), kind, lhs, rhs);
    } else if (!LHSTy->isAnyComplexType() && !RHSTy->isAnyComplexType()) {
      BinOpInfo BOInfo = emitBinOps(E);
      mlir::Value LHS = BOInfo.LHS;
      mlir::Value RHS = BOInfo.RHS;

      if (LHSTy->isVectorType()) {
        if (!E->getType()->isVectorType()) {
          // If AltiVec, the comparison results in a numeric type, so we use
          // intrinsics comparing vectors and giving 0 or 1 as a result
          llvm_unreachable("NYI: AltiVec comparison");
        } else {
          // Other kinds of vectors.  Element-wise comparison returning
          // a vector.
          cir::CmpOpKind Kind = ClangCmpToCIRCmp(E->getOpcode());
          Result = Builder.create<cir::VecCmpOp>(
              CGF.getLoc(BOInfo.Loc), CGF.convertType(BOInfo.FullType), Kind,
              BOInfo.LHS, BOInfo.RHS);
        }
      } else if (BOInfo.isFixedPointOp()) {
        assert(0 && "not implemented");
      } else {
        // FIXME(cir): handle another if above for CIR equivalent on
        // LHSTy->hasSignedIntegerRepresentation()

        // Unsigned integers and pointers.
        if (CGF.CGM.getCodeGenOpts().StrictVTablePointers &&
            mlir::isa<cir::PointerType>(LHS.getType()) &&
            mlir::isa<cir::PointerType>(RHS.getType())) {
          llvm_unreachable("NYI");
        }

        cir::CmpOpKind Kind = ClangCmpToCIRCmp(E->getOpcode());
        Result = Builder.createCompare(CGF.getLoc(BOInfo.Loc), Kind, LHS, RHS);
      }
    } else { // Complex Comparison: can only be an equality comparison.
      assert(0 && "not implemented");
    }

    return emitScalarConversion(Result, CGF.getContext().BoolTy, E->getType(),
                                E->getExprLoc());
  }

  mlir::Value emitFloatToBoolConversion(mlir::Value src, mlir::Location loc) {
    auto boolTy = Builder.getBoolTy();
    return Builder.create<cir::CastOp>(loc, boolTy,
                                       cir::CastKind::float_to_bool, src);
  }

  mlir::Value emitIntToBoolConversion(mlir::Value srcVal, mlir::Location loc) {
    // Because of the type rules of C, we often end up computing a
    // logical value, then zero extending it to int, then wanting it
    // as a logical value again.
    // TODO: optimize this common case here or leave it for later
    // CIR passes?
    mlir::Type boolTy = CGF.convertType(CGF.getContext().BoolTy);
    return Builder.create<cir::CastOp>(loc, boolTy, cir::CastKind::int_to_bool,
                                       srcVal);
  }

  /// Convert the specified expression value to a boolean (!cir.bool) truth
  /// value. This is equivalent to "Val != 0".
  mlir::Value emitConversionToBool(mlir::Value Src, QualType SrcType,
                                   mlir::Location loc) {
    assert(SrcType.isCanonical() && "EmitScalarConversion strips typedefs");

    if (SrcType->isRealFloatingType())
      return emitFloatToBoolConversion(Src, loc);

    if (auto *MPT = llvm::dyn_cast<MemberPointerType>(SrcType))
      assert(0 && "not implemented");

    if (SrcType->isIntegerType())
      return emitIntToBoolConversion(Src, loc);

    assert(::mlir::isa<cir::PointerType>(Src.getType()));
    return emitPointerToBoolConversion(Src, SrcType);
  }

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  /// TODO: do we need ScalarConversionOpts here? Should be done in another
  /// pass.
  mlir::Value
  emitScalarConversion(mlir::Value Src, QualType SrcType, QualType DstType,
                       SourceLocation Loc,
                       ScalarConversionOpts Opts = ScalarConversionOpts()) {
    // All conversions involving fixed point types should be handled by the
    // emitFixedPoint family functions. This is done to prevent bloating up
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
      return emitConversionToBool(Src, SrcType, CGF.getLoc(Loc));

    mlir::Type DstTy = convertType(DstType);

    // Cast from half through float if half isn't a native type.
    if (SrcType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      // Cast to FP using the intrinsic if the half type itself isn't supported.
      if (mlir::isa<cir::CIRFPTypeInterface>(DstTy)) {
        if (CGF.getContext().getTargetInfo().useFP16ConversionIntrinsics())
          llvm_unreachable("cast via llvm.convert.from.fp16 is NYI");
      } else {
        // Cast to other types through float, using either the intrinsic or
        // FPExt, depending on whether the half type itself is supported (as
        // opposed to operations on half, available with NativeHalfType).
        if (CGF.getContext().getTargetInfo().useFP16ConversionIntrinsics()) {
          llvm_unreachable("cast via llvm.convert.from.fp16 is NYI");
        } else {
          Src = Builder.createCast(CGF.getLoc(Loc), cir::CastKind::floating,
                                   Src, CGF.FloatTy);
        }
        SrcType = CGF.getContext().FloatTy;
        SrcTy = CGF.FloatTy;
      }
    }

    // TODO(cir): LLVM codegen ignore conversions like int -> uint,
    // is there anything to be done for CIR here?
    if (SrcTy == DstTy) {
      if (Opts.EmitImplicitIntegerSignChangeChecks)
        llvm_unreachable("not implemented");
      return Src;
    }

    // Handle pointer conversions next: pointers can only be converted to/from
    // other pointers and integers. Check for pointer types in terms of LLVM, as
    // some native types (like Obj-C id) may map to a pointer type.
    if (auto DstPT = dyn_cast<cir::PointerType>(DstTy)) {
      llvm_unreachable("NYI");
    }

    if (isa<cir::PointerType>(SrcTy)) {
      // Must be an ptr to int cast.
      assert(isa<cir::IntType>(DstTy) && "not ptr->int?");
      return Builder.createPtrToInt(Src, DstTy);
    }

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

    // Finally, we have the arithmetic types or vectors of arithmetic types.
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
      // Make sure we cast in a single step if from another FP type.
      if (mlir::isa<cir::CIRFPTypeInterface>(SrcTy)) {
        // Use the intrinsic if the half type itself isn't supported
        // (as opposed to operations on half, available with NativeHalfType).
        if (CGF.getContext().getTargetInfo().useFP16ConversionIntrinsics())
          llvm_unreachable("cast via llvm.convert.to.fp16 is NYI");
        // If the half type is supported, just use an fptrunc.
        return Builder.createCast(CGF.getLoc(Loc), cir::CastKind::floating, Src,
                                  DstTy);
      }
      DstTy = CGF.FloatTy;
    }

    Res = emitScalarCast(Src, SrcType, DstType, SrcTy, DstTy, Opts);

    if (DstTy != ResTy) {
      if (CGF.getContext().getTargetInfo().useFP16ConversionIntrinsics()) {
        llvm_unreachable("cast via llvm.convert.to.fp16 is NYI");
      } else {
        Res = Builder.createCast(CGF.getLoc(Loc), cir::CastKind::floating, Res,
                                 ResTy);
      }
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
mlir::Value CIRGenFunction::emitScalarExpr(const Expr *E) {
  assert(E && hasScalarEvaluationKind(E->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(E));
}

mlir::Value CIRGenFunction::emitPromotedScalarExpr(const Expr *E,
                                                   QualType PromotionType) {
  if (!PromotionType.isNull())
    return ScalarExprEmitter(*this, builder).emitPromoted(E, PromotionType);
  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(E));
}

[[maybe_unused]] static bool MustVisitNullValue(const Expr *E) {
  // If a null pointer expression's type is the C++0x nullptr_t, then
  // it's not necessarily a simple constant and it must be evaluated
  // for its potential side effects.
  return E->getType()->isNullPtrType();
}

/// If \p E is a widened promoted integer, get its base (unpromoted) type.
static std::optional<QualType>
getUnwidenedIntegerType(const ASTContext &astContext, const Expr *E) {
  const Expr *Base = E->IgnoreImpCasts();
  if (E == Base)
    return std::nullopt;

  QualType BaseTy = Base->getType();
  if (!astContext.isPromotableIntegerType(BaseTy) ||
      astContext.getTypeSize(BaseTy) >= astContext.getTypeSize(E->getType()))
    return std::nullopt;

  return BaseTy;
}

/// Check if \p E is a widened promoted integer.
[[maybe_unused]] static bool IsWidenedIntegerOp(const ASTContext &astContext,
                                                const Expr *E) {
  return getUnwidenedIntegerType(astContext, E).has_value();
}

/// Check if we can skip the overflow check for \p Op.
[[maybe_unused]] static bool CanElideOverflowCheck(const ASTContext &astContext,
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
  auto OptionalLHSTy = getUnwidenedIntegerType(astContext, BO->getLHS());
  if (!OptionalLHSTy)
    return false;

  auto OptionalRHSTy = getUnwidenedIntegerType(astContext, BO->getRHS());
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
  unsigned PromotedSize = astContext.getTypeSize(Op.E->getType());
  return (2 * astContext.getTypeSize(LHSTy)) < PromotedSize ||
         (2 * astContext.getTypeSize(RHSTy)) < PromotedSize;
}

/// Emit pointer + index arithmetic.
static mlir::Value emitPointerArithmetic(CIRGenFunction &CGF,
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
  if (!isSubtraction && !mlir::isa<cir::PointerType>(pointer.getType())) {
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
    return CGF.getBuilder().createIntToPtr(index, pointer.getType());

  // Differently from LLVM codegen, ABI bits for index sizes is handled during
  // LLVM lowering.

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
          CGF.getContext().getAsVariableArrayType(elementType)) {

    // The element count here is the total number of non-VLA elements.
    mlir::Value numElements = CGF.getVLASize(vla).NumElts;

    // GEP indexes are signed, and scaling an index isn't permitted to
    // signed-overflow, so we use the same semantics for our explicit
    // multiply.  We suppress this if overflow is not undefined behavior.
    mlir::Type elemTy = CGF.convertTypeForMem(vla->getElementType());

    index = CGF.getBuilder().createCast(cir::CastKind::integral, index,
                                        numElements.getType());
    index = CGF.getBuilder().createMul(index, numElements);

    if (CGF.getLangOpts().isSignedOverflowDefined()) {
      pointer = CGF.getBuilder().create<cir::PtrStrideOp>(
          CGF.getLoc(op.E->getExprLoc()), pointer.getType(), pointer, index);
    } else {
      pointer = CGF.emitCheckedInBoundsGEP(elemTy, pointer, index, isSigned,
                                           isSubtraction, op.E->getExprLoc());
    }
    return pointer;
  }
  // Explicitly handle GNU void* and function pointer arithmetic extensions. The
  // GNU void* casts amount to no-ops since our void* type is i8*, but this is
  // future proof.
  mlir::Type elemTy;
  if (elementType->isVoidType() || elementType->isFunctionType())
    elemTy = CGF.UInt8Ty;
  else
    elemTy = CGF.convertTypeForMem(elementType);

  if (CGF.getLangOpts().isSignedOverflowDefined())
    return CGF.getBuilder().create<cir::PtrStrideOp>(
        CGF.getLoc(op.E->getExprLoc()), pointer.getType(), pointer, index);

  return CGF.emitCheckedInBoundsGEP(elemTy, pointer, index, isSigned,
                                    isSubtraction, op.E->getExprLoc());
}

mlir::Value ScalarExprEmitter::emitMul(const BinOpInfo &Ops) {
  if (Ops.CompType->isSignedIntegerOrEnumerationType()) {
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createMul(Ops.LHS, Ops.RHS);
      [[fallthrough]];
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createNSWMul(Ops.LHS, Ops.RHS);
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      if (CanElideOverflowCheck(CGF.getContext(), Ops))
        return Builder.createNSWMul(Ops.LHS, Ops.RHS);
      llvm_unreachable("NYI");
    }
  }
  if (Ops.FullType->isConstantMatrixType()) {
    llvm_unreachable("NYI");
  }
  if (Ops.CompType->isUnsignedIntegerType() &&
      CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
      !CanElideOverflowCheck(CGF.getContext(), Ops))
    llvm_unreachable("NYI");

  if (cir::isFPOrFPVectorTy(Ops.LHS.getType())) {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, Ops.FPFeatures);
    return Builder.createFMul(Ops.LHS, Ops.RHS);
  }

  if (Ops.isFixedPointOp())
    llvm_unreachable("NYI");

  return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                    CGF.convertType(Ops.FullType),
                                    cir::BinOpKind::Mul, Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::emitDiv(const BinOpInfo &Ops) {
  return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                    CGF.convertType(Ops.FullType),
                                    cir::BinOpKind::Div, Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::emitRem(const BinOpInfo &Ops) {
  return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                    CGF.convertType(Ops.FullType),
                                    cir::BinOpKind::Rem, Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::emitAdd(const BinOpInfo &Ops) {
  if (mlir::isa<cir::PointerType>(Ops.LHS.getType()) ||
      mlir::isa<cir::PointerType>(Ops.RHS.getType()))
    return emitPointerArithmetic(CGF, Ops, /*isSubtraction=*/false);
  if (Ops.CompType->isSignedIntegerOrEnumerationType()) {
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createAdd(Ops.LHS, Ops.RHS);
      [[fallthrough]];
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createNSWAdd(Ops.LHS, Ops.RHS);
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      if (CanElideOverflowCheck(CGF.getContext(), Ops))
        return Builder.createNSWAdd(Ops.LHS, Ops.RHS);

      llvm_unreachable("NYI");
    }
  }
  if (Ops.FullType->isConstantMatrixType()) {
    llvm_unreachable("NYI");
  }

  if (Ops.CompType->isUnsignedIntegerType() &&
      CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
      !CanElideOverflowCheck(CGF.getContext(), Ops))
    llvm_unreachable("NYI");

  if (cir::isFPOrFPVectorTy(Ops.LHS.getType())) {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, Ops.FPFeatures);
    return Builder.createFAdd(Ops.LHS, Ops.RHS);
  }

  if (Ops.isFixedPointOp())
    llvm_unreachable("NYI");

  return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                    CGF.convertType(Ops.FullType),
                                    cir::BinOpKind::Add, Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::emitSub(const BinOpInfo &Ops) {
  // The LHS is always a pointer if either side is.
  if (!mlir::isa<cir::PointerType>(Ops.LHS.getType())) {
    if (Ops.CompType->isSignedIntegerOrEnumerationType()) {
      switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
      case LangOptions::SOB_Defined: {
        if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return Builder.createSub(Ops.LHS, Ops.RHS);
        [[fallthrough]];
      }
      case LangOptions::SOB_Undefined:
        if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return Builder.createNSWSub(Ops.LHS, Ops.RHS);
        [[fallthrough]];
      case LangOptions::SOB_Trapping:
        if (CanElideOverflowCheck(CGF.getContext(), Ops))
          return Builder.createNSWSub(Ops.LHS, Ops.RHS);
        llvm_unreachable("NYI");
      }
    }

    if (Ops.FullType->isConstantMatrixType()) {
      llvm_unreachable("NYI");
    }

    if (Ops.CompType->isUnsignedIntegerType() &&
        CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
        !CanElideOverflowCheck(CGF.getContext(), Ops))
      llvm_unreachable("NYI");

    if (cir::isFPOrFPVectorTy(Ops.LHS.getType())) {
      CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, Ops.FPFeatures);
      return Builder.createFSub(Ops.LHS, Ops.RHS);
    }

    if (Ops.isFixedPointOp())
      llvm_unreachable("NYI");

    return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                      CGF.convertType(Ops.FullType),
                                      cir::BinOpKind::Sub, Ops.LHS, Ops.RHS);
  }

  // If the RHS is not a pointer, then we have normal pointer
  // arithmetic.
  if (!mlir::isa<cir::PointerType>(Ops.RHS.getType()))
    return emitPointerArithmetic(CGF, Ops, /*isSubtraction=*/true);

  // Otherwise, this is a pointer subtraction

  // Do the raw subtraction part.
  //
  // TODO(cir): note for LLVM lowering out of this; when expanding this into
  // LLVM we shall take VLA's, division by element size, etc.
  //
  // See more in `EmitSub` in CGExprScalar.cpp.
  assert(!cir::MissingFeatures::llvmLoweringPtrDiffConsidersPointee());
  return Builder.create<cir::PtrDiffOp>(CGF.getLoc(Ops.Loc), CGF.PtrDiffTy,
                                        Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::emitShl(const BinOpInfo &Ops) {
  // TODO: This misses out on the sanitizer check below.
  if (Ops.isFixedPointOp())
    llvm_unreachable("NYI");

  // CIR accepts shift between different types, meaning nothing special
  // to be done here. OTOH, LLVM requires the LHS and RHS to be the same type:
  // promote or truncate the RHS to the same size as the LHS.

  bool SanitizeSignedBase = CGF.SanOpts.has(SanitizerKind::ShiftBase) &&
                            Ops.CompType->hasSignedIntegerRepresentation() &&
                            !CGF.getLangOpts().isSignedOverflowDefined() &&
                            !CGF.getLangOpts().CPlusPlus20;
  bool SanitizeUnsignedBase =
      CGF.SanOpts.has(SanitizerKind::UnsignedShiftBase) &&
      Ops.CompType->hasUnsignedIntegerRepresentation();
  bool SanitizeBase = SanitizeSignedBase || SanitizeUnsignedBase;
  bool SanitizeExponent = CGF.SanOpts.has(SanitizerKind::ShiftExponent);

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (CGF.getLangOpts().OpenCL)
    llvm_unreachable("NYI");
  else if ((SanitizeBase || SanitizeExponent) &&
           mlir::isa<cir::IntType>(Ops.LHS.getType())) {
    llvm_unreachable("NYI");
  }

  return Builder.create<cir::ShiftOp>(CGF.getLoc(Ops.Loc),
                                      CGF.convertType(Ops.FullType), Ops.LHS,
                                      Ops.RHS, CGF.getBuilder().getUnitAttr());
}

mlir::Value ScalarExprEmitter::emitShr(const BinOpInfo &Ops) {
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
           mlir::isa<cir::IntType>(Ops.LHS.getType())) {
    llvm_unreachable("NYI");
  }

  // Note that we don't need to distinguish unsigned treatment at this
  // point since it will be handled later by LLVM lowering.
  return Builder.create<cir::ShiftOp>(
      CGF.getLoc(Ops.Loc), CGF.convertType(Ops.FullType), Ops.LHS, Ops.RHS);
}

mlir::Value ScalarExprEmitter::emitAnd(const BinOpInfo &Ops) {
  return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                    CGF.convertType(Ops.FullType),
                                    cir::BinOpKind::And, Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::emitXor(const BinOpInfo &Ops) {
  return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                    CGF.convertType(Ops.FullType),
                                    cir::BinOpKind::Xor, Ops.LHS, Ops.RHS);
}
mlir::Value ScalarExprEmitter::emitOr(const BinOpInfo &Ops) {
  return Builder.create<cir::BinOp>(CGF.getLoc(Ops.Loc),
                                    CGF.convertType(Ops.FullType),
                                    cir::BinOpKind::Or, Ops.LHS, Ops.RHS);
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
  case CK_ObjCObjectLValueCast:
  case CK_LValueToRValueBitCast: {
    LValue SourceLVal = CGF.emitLValue(E);
    Address SourceAddr = SourceLVal.getAddress();

    mlir::Type DestElemTy = CGF.convertTypeForMem(DestTy);
    mlir::Type DestPtrTy = CGF.getBuilder().getPointerTo(DestElemTy);
    mlir::Value DestPtr = CGF.getBuilder().createBitcast(
        CGF.getLoc(E->getExprLoc()), SourceAddr.getPointer(), DestPtrTy);

    Address DestAddr = Address(DestPtr, DestElemTy, SourceAddr.getAlignment(),
                               SourceAddr.isKnownNonNull());
    LValue DestLVal = CGF.makeAddrLValue(DestAddr, DestTy);
    DestLVal.setTBAAInfo(TBAAAccessInfo::getMayAliasInfo());
    return emitLoadOfLValue(DestLVal, CE->getExprLoc());
  }

  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_BitCast: {
    auto Src = Visit(const_cast<Expr *>(E));
    mlir::Type DstTy = CGF.convertType(DestTy);

    assert(!cir::MissingFeatures::addressSpace());
    if (CGF.SanOpts.has(SanitizerKind::CFIUnrelatedCast)) {
      llvm_unreachable("NYI");
    }

    if (CGF.CGM.getCodeGenOpts().StrictVTablePointers) {
      llvm_unreachable("NYI");
    }

    // Update heapallocsite metadata when there is an explicit pointer cast.
    assert(!cir::MissingFeatures::addHeapAllocSiteMetadata());

    // If Src is a fixed vector and Dst is a scalable vector, and both have the
    // same element type, use the llvm.vector.insert intrinsic to perform the
    // bitcast.
    assert(!cir::MissingFeatures::scalableVectors());

    // If Src is a scalable vector and Dst is a fixed vector, and both have the
    // same element type, use the llvm.vector.extract intrinsic to perform the
    // bitcast.
    assert(!cir::MissingFeatures::scalableVectors());

    // Perform VLAT <-> VLST bitcast through memory.
    // TODO: since the llvm.experimental.vector.{insert,extract} intrinsics
    //       require the element types of the vectors to be the same, we
    //       need to keep this around for bitcasts between VLAT <-> VLST where
    //       the element types of the vectors are not the same, until we figure
    //       out a better way of doing these casts.
    assert(!cir::MissingFeatures::scalableVectors());

    return CGF.getBuilder().createBitcast(CGF.getLoc(E->getSourceRange()), Src,
                                          DstTy);
  }
  case CK_AddressSpaceConversion: {
    Expr::EvalResult Result;
    if (E->EvaluateAsRValue(Result, CGF.getContext()) &&
        Result.Val.isNullPointer()) {
      // If E has side effect, it is emitted even if its final result is a
      // null pointer. In that case, a DCE pass should be able to
      // eliminate the useless instructions emitted during translating E.
      if (Result.HasSideEffects)
        Visit(E);
      return CGF.CGM.emitNullConstant(DestTy, CGF.getLoc(E->getExprLoc()));
    }
    // Since target may map different address spaces in AST to the same address
    // space, an address space conversion may end up as a bitcast.
    auto SrcAS = CGF.builder.getAddrSpaceAttr(
        E->getType()->getPointeeType().getAddressSpace());
    auto DestAS = CGF.builder.getAddrSpaceAttr(
        DestTy->getPointeeType().getAddressSpace());
    return CGF.CGM.getTargetCIRGenInfo().performAddrSpaceCast(
        CGF, Visit(E), SrcAS, DestAS, convertType(DestTy));
  }
  case CK_AtomicToNonAtomic:
    llvm_unreachable("NYI");
  case CK_NonAtomicToAtomic:
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
  case CK_BaseToDerived: {
    const CXXRecordDecl *DerivedClassDecl = DestTy->getPointeeCXXRecordDecl();
    assert(DerivedClassDecl && "BaseToDerived arg isn't a C++ object pointer!");
    Address Base = CGF.emitPointerWithAlignment(E);
    Address Derived = CGF.getAddressOfDerivedClass(
        Base, DerivedClassDecl, CE->path_begin(), CE->path_end(),
        CGF.shouldNullCheckClassCastValue(CE));

    // C++11 [expr.static.cast]p11: Behavior is undefined if a downcast is
    // performed and the object is not of the derived type.
    if (CGF.sanitizePerformTypeCheck())
      assert(!cir::MissingFeatures::sanitizeOther());

    if (CGF.SanOpts.has(SanitizerKind::CFIDerivedCast))
      assert(!cir::MissingFeatures::sanitizeOther());

    return CGF.getAsNaturalPointerTo(Derived, CE->getType()->getPointeeType());
  }
  case CK_DerivedToBase: {
    // The EmitPointerWithAlignment path does this fine; just discard
    // the alignment.
    return CGF.emitPointerWithAlignment(CE).getPointer();
  }
  case CK_Dynamic: {
    Address V = CGF.emitPointerWithAlignment(E);
    const auto *DCE = cast<CXXDynamicCastExpr>(CE);
    return CGF.emitDynamicCast(V, DCE);
  }
  case CK_ArrayToPointerDecay:
    return CGF.emitArrayToPointerDecay(E).getPointer();
  case CK_FunctionToPointerDecay:
    return emitLValue(E).getPointer();

  case CK_NullToPointer: {
    // FIXME: use MustVisitNullValue(E) and evaluate expr.
    // Note that DestTy is used as the MLIR type instead of a custom
    // nullptr type.
    mlir::Type Ty = CGF.convertType(DestTy);
    return Builder.getNullPtr(Ty, CGF.getLoc(E->getExprLoc()));
  }

  case CK_NullToMemberPointer: {
    if (MustVisitNullValue(E))
      CGF.emitIgnoredExpr(E);

    assert(!cir::MissingFeatures::cxxABI());

    const MemberPointerType *MPT = CE->getType()->getAs<MemberPointerType>();
    if (MPT->isMemberFunctionPointerType()) {
      auto Ty = mlir::cast<cir::MethodType>(CGF.convertType(DestTy));
      return Builder.getNullMethodPtr(Ty, CGF.getLoc(E->getExprLoc()));
    }

    auto Ty = mlir::cast<cir::DataMemberType>(CGF.convertType(DestTy));
    return Builder.getNullDataMemberPtr(Ty, CGF.getLoc(E->getExprLoc()));
  }
  case CK_ReinterpretMemberPointer: {
    mlir::Value src = Visit(E);
    return Builder.createBitcast(CGF.getLoc(E->getExprLoc()), src,
                                 CGF.convertType(DestTy));
  }
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer: {
    mlir::Value src = Visit(E);

    if (E->getType()->isMemberFunctionPointerType())
      assert(!cir::MissingFeatures::memberFuncPtrAuthInfo());

    QualType derivedTy =
        Kind == CK_DerivedToBaseMemberPointer ? E->getType() : CE->getType();
    const CXXRecordDecl *derivedClass = derivedTy->castAs<MemberPointerType>()
                                            ->getClass()
                                            ->getAsCXXRecordDecl();
    CharUnits offset = CGF.CGM.computeNonVirtualBaseClassOffset(
        derivedClass, CE->path_begin(), CE->path_end());

    mlir::Location loc = CGF.getLoc(E->getExprLoc());
    mlir::Type resultTy = CGF.convertType(DestTy);
    mlir::IntegerAttr offsetAttr = Builder.getIndexAttr(offset.getQuantity());

    if (E->getType()->isMemberFunctionPointerType()) {
      if (Kind == CK_BaseToDerivedMemberPointer)
        return Builder.create<cir::DerivedMethodOp>(loc, resultTy, src,
                                                    offsetAttr);
      return Builder.create<cir::BaseMethodOp>(loc, resultTy, src, offsetAttr);
    }

    if (Kind == CK_BaseToDerivedMemberPointer)
      return Builder.create<cir::DerivedDataMemberOp>(loc, resultTy, src,
                                                      offsetAttr);
    return Builder.create<cir::BaseDataMemberOp>(loc, resultTy, src,
                                                 offsetAttr);
  }
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
  case CK_FloatingComplexCast:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_FloatingComplexToIntegralComplex:
    llvm_unreachable("scalar cast to non-scalar value");

  case CK_ConstructorConversion:
    llvm_unreachable("NYI");
  case CK_ToUnion:
    llvm_unreachable("NYI");

  case CK_LValueToRValue:
    assert(CGF.getContext().hasSameUnqualifiedType(E->getType(), DestTy));
    assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
    return Visit(const_cast<Expr *>(E));

  case CK_IntegralToPointer: {
    auto DestCIRTy = convertType(DestTy);
    mlir::Value Src = Visit(const_cast<Expr *>(E));

    // Properly resize by casting to an int of the same size as the pointer.
    // Clang's IntegralToPointer includes 'bool' as the source, but in CIR
    // 'bool' is not an integral type.  So check the source type to get the
    // correct CIR conversion.
    auto MiddleTy = CGF.CGM.getDataLayout().getIntPtrType(DestCIRTy);
    auto MiddleVal = Builder.createCast(E->getType()->isBooleanType()
                                            ? cir::CastKind::bool_to_int
                                            : cir::CastKind::integral,
                                        Src, MiddleTy);

    if (CGF.CGM.getCodeGenOpts().StrictVTablePointers)
      llvm_unreachable("NYI");

    return Builder.createIntToPtr(MiddleVal, DestCIRTy);
  }
  case CK_PointerToIntegral: {
    assert(!DestTy->isBooleanType() && "bool should use PointerToBool");
    if (CGF.CGM.getCodeGenOpts().StrictVTablePointers)
      llvm_unreachable("NYI");
    return Builder.createPtrToInt(Visit(E), convertType(DestTy));
  }
  case CK_ToVoid: {
    CGF.emitIgnoredExpr(E);
    return nullptr;
  }
  case CK_MatrixCast:
    llvm_unreachable("NYI");
  case CK_VectorSplat: {
    // Create a vector object and fill all elements with the same scalar value.
    assert(DestTy->isVectorType() && "CK_VectorSplat to non-vector type");
    return CGF.getBuilder().create<cir::VecSplatOp>(
        CGF.getLoc(E->getSourceRange()), CGF.convertType(DestTy), Visit(E));
  }
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
    return emitScalarConversion(Visit(E), E->getType(), DestTy,
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
    return emitScalarConversion(Visit(E), E->getType(), DestTy,
                                CE->getExprLoc());
  }
  case CK_BooleanToSignedIntegral:
    llvm_unreachable("NYI");

  case CK_IntegralToBoolean: {
    return emitIntToBoolConversion(Visit(E), CGF.getLoc(CE->getSourceRange()));
  }

  case CK_PointerToBoolean:
    return emitPointerToBoolConversion(Visit(E), E->getType());
  case CK_FloatingToBoolean:
    return emitFloatToBoolConversion(Visit(E), CGF.getLoc(E->getExprLoc()));
  case CK_MemberPointerToBoolean: {
    mlir::Value memPtr = Visit(E);
    return Builder.createCast(CGF.getLoc(CE->getSourceRange()),
                              cir::CastKind::member_ptr_to_bool, memPtr,
                              CGF.convertType(DestTy));
  }
  case CK_FloatingComplexToReal:
  case CK_IntegralComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToBoolean: {
    mlir::Value V = CGF.emitComplexExpr(E);
    return emitComplexToScalarConversion(CGF.getLoc(CE->getExprLoc()), V, Kind,
                                         DestTy);
  }
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
    return emitLoadOfLValue(E);

  auto V = CGF.emitCallExpr(E).getScalarVal();
  assert(!cir::MissingFeatures::emitLValueAlignmentAssumption());
  return V;
}

mlir::Value ScalarExprEmitter::VisitMemberExpr(MemberExpr *E) {
  // TODO(cir): Folding all this constants sound like work for MLIR optimizers,
  // keep assertion for now.
  assert(!cir::MissingFeatures::tryEmitAsConstant());
  Expr::EvalResult Result;
  if (E->EvaluateAsInt(Result, CGF.getContext(), Expr::SE_AllowSideEffects)) {
    llvm::APSInt Value = Result.Val.getInt();
    CGF.emitIgnoredExpr(E->getBase());
    return Builder.getConstInt(CGF.getLoc(E->getExprLoc()), Value);
  }
  return emitLoadOfLValue(E);
}

/// Emit a conversion from the specified type to the specified destination
/// type, both of which are CIR scalar types.
mlir::Value CIRGenFunction::emitScalarConversion(mlir::Value Src,
                                                 QualType SrcTy, QualType DstTy,
                                                 SourceLocation Loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(SrcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(DstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this, builder)
      .emitScalarConversion(Src, SrcTy, DstTy, Loc);
}

mlir::Value CIRGenFunction::emitComplexToScalarConversion(mlir::Value Src,
                                                          QualType SrcTy,
                                                          QualType DstTy,
                                                          SourceLocation Loc) {
  assert(SrcTy->isAnyComplexType() && hasScalarEvaluationKind(DstTy) &&
         "Invalid complex -> scalar conversion");

  auto ComplexElemTy = SrcTy->castAs<ComplexType>()->getElementType();
  if (DstTy->isBooleanType()) {
    auto Kind = ComplexElemTy->isFloatingType()
                    ? cir::CastKind::float_complex_to_bool
                    : cir::CastKind::int_complex_to_bool;
    return builder.createCast(getLoc(Loc), Kind, Src, convertType(DstTy));
  }

  auto Kind = ComplexElemTy->isFloatingType()
                  ? cir::CastKind::float_complex_to_real
                  : cir::CastKind::int_complex_to_real;
  auto Real =
      builder.createCast(getLoc(Loc), Kind, Src, convertType(ComplexElemTy));
  return emitScalarConversion(Real, ComplexElemTy, DstTy, Loc);
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

  if (E->getType()->isVectorType()) {
    assert(!cir::MissingFeatures::scalableVectors() &&
           "NYI: scalable vector init");
    assert(!cir::MissingFeatures::vectorConstants() && "NYI: vector constants");
    auto VectorType =
        mlir::dyn_cast<cir::VectorType>(CGF.convertType(E->getType()));
    SmallVector<mlir::Value, 16> Elements;
    for (Expr *init : E->inits()) {
      Elements.push_back(Visit(init));
    }
    // Zero-initialize any remaining values.
    if (NumInitElements < VectorType.getSize()) {
      mlir::Value ZeroValue = CGF.getBuilder().create<cir::ConstantOp>(
          CGF.getLoc(E->getSourceRange()), VectorType.getEltType(),
          CGF.getBuilder().getZeroInitAttr(VectorType.getEltType()));
      for (uint64_t i = NumInitElements; i < VectorType.getSize(); ++i) {
        Elements.push_back(ZeroValue);
      }
    }
    return CGF.getBuilder().create<cir::VecCreateOp>(
        CGF.getLoc(E->getSourceRange()), VectorType, Elements);
  }

  if (NumInitElements == 0) {
    // C++11 value-initialization for the scalar.
    return emitNullValue(E->getType(), CGF.getLoc(E->getExprLoc()));
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
  auto dstTy = convertType(E->getType());
  if (mlir::isa<cir::IntType>(dstTy))
    return Builder.createBoolToInt(boolVal, dstTy);
  if (mlir::isa<cir::BoolType>(dstTy))
    return boolVal;

  llvm_unreachable("destination type for logical-not unary operator is NYI");
}

mlir::Value ScalarExprEmitter::VisitReal(const UnaryOperator *E) {
  // TODO(cir): handle scalar promotion.

  Expr *Op = E->getSubExpr();
  if (Op->getType()->isAnyComplexType()) {
    // If it's an l-value, load through the appropriate subobject l-value.
    // Note that we have to ask E because Op might be an l-value that
    // this won't work for, e.g. an Obj-C property.
    if (E->isGLValue())
      return CGF.emitLoadOfLValue(CGF.emitLValue(E), E->getExprLoc())
          .getScalarVal();
    // Otherwise, calculate and project.
    llvm_unreachable("NYI");
  }

  return Visit(Op);
}

mlir::Value ScalarExprEmitter::VisitImag(const UnaryOperator *E) {
  // TODO(cir): handle scalar promotion.

  Expr *Op = E->getSubExpr();
  if (Op->getType()->isAnyComplexType()) {
    // If it's an l-value, load through the appropriate subobject l-value.
    // Note that we have to ask E because Op might be an l-value that
    // this won't work for, e.g. an Obj-C property.
    if (E->isGLValue())
      return CGF.emitLoadOfLValue(CGF.emitLValue(E), E->getExprLoc())
          .getScalarVal();
    // Otherwise, calculate and project.
    llvm_unreachable("NYI");
  }

  return Visit(Op);
}

// Conversion from bool, integral, or floating-point to integral or
// floating-point. Conversions involving other types are handled elsewhere.
// Conversion to bool is handled elsewhere because that's a comparison against
// zero, not a simple cast. This handles both individual scalars and vectors.
mlir::Value ScalarExprEmitter::emitScalarCast(mlir::Value Src, QualType SrcType,
                                              QualType DstType,
                                              mlir::Type SrcTy,
                                              mlir::Type DstTy,
                                              ScalarConversionOpts Opts) {
  assert(!SrcType->isMatrixType() && !DstType->isMatrixType() &&
         "Internal error: matrix types not handled by this function.");
  if (mlir::isa<mlir::IntegerType>(SrcTy) ||
      mlir::isa<mlir::IntegerType>(DstTy))
    llvm_unreachable("Obsolete code. Don't use mlir::IntegerType with CIR.");

  mlir::Type FullDstTy = DstTy;
  if (mlir::isa<cir::VectorType>(SrcTy) && mlir::isa<cir::VectorType>(DstTy)) {
    // Use the element types of the vectors to figure out the CastKind.
    SrcTy = mlir::dyn_cast<cir::VectorType>(SrcTy).getEltType();
    DstTy = mlir::dyn_cast<cir::VectorType>(DstTy).getEltType();
  }
  assert(!mlir::isa<cir::VectorType>(SrcTy) &&
         !mlir::isa<cir::VectorType>(DstTy) &&
         "emitScalarCast given a vector type and a non-vector type");

  std::optional<cir::CastKind> CastKind;

  if (mlir::isa<cir::BoolType>(SrcTy)) {
    if (Opts.TreatBooleanAsSigned)
      llvm_unreachable("NYI: signed bool");
    if (CGF.getBuilder().isInt(DstTy)) {
      CastKind = cir::CastKind::bool_to_int;
    } else if (mlir::isa<cir::CIRFPTypeInterface>(DstTy)) {
      CastKind = cir::CastKind::bool_to_float;
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else if (CGF.getBuilder().isInt(SrcTy)) {
    if (CGF.getBuilder().isInt(DstTy)) {
      CastKind = cir::CastKind::integral;
    } else if (mlir::isa<cir::CIRFPTypeInterface>(DstTy)) {
      CastKind = cir::CastKind::int_to_float;
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else if (mlir::isa<cir::CIRFPTypeInterface>(SrcTy)) {
    if (CGF.getBuilder().isInt(DstTy)) {
      // If we can't recognize overflow as undefined behavior, assume that
      // overflow saturates. This protects against normal optimizations if we
      // are compiling with non-standard FP semantics.
      if (!CGF.CGM.getCodeGenOpts().StrictFloatCastOverflow)
        llvm_unreachable("NYI");
      if (Builder.getIsFPConstrained())
        llvm_unreachable("NYI");
      CastKind = cir::CastKind::float_to_int;
    } else if (mlir::isa<cir::CIRFPTypeInterface>(DstTy)) {
      // TODO: split this to createFPExt/createFPTrunc
      return Builder.createFloatingCast(Src, FullDstTy);
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else {
    llvm_unreachable("Internal error: Cast from unexpected type");
  }

  assert(CastKind.has_value() && "Internal error: CastKind not set.");
  return Builder.create<cir::CastOp>(Src.getLoc(), FullDstTy, *CastKind, Src);
}

LValue
CIRGenFunction::emitCompoundAssignmentLValue(const CompoundAssignOperator *E) {
  ScalarExprEmitter Scalar(*this, builder);
  mlir::Value Result;
  switch (E->getOpcode()) {
#define COMPOUND_OP(Op)                                                        \
  case BO_##Op##Assign:                                                        \
    return Scalar.emitCompoundAssignLValue(E, &ScalarExprEmitter::emit##Op,    \
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

LValue ScalarExprEmitter::emitCompoundAssignLValue(
    const CompoundAssignOperator *E,
    mlir::Value (ScalarExprEmitter::*Func)(const BinOpInfo &),
    mlir::Value &Result) {
  QualType LHSTy = E->getLHS()->getType();
  BinOpInfo OpInfo;

  if (E->getComputationResultType()->isAnyComplexType())
    assert(0 && "not implemented");

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.

  QualType PromotionTypeCR = getPromotionType(E->getComputationResultType());
  if (PromotionTypeCR.isNull())
    PromotionTypeCR = E->getComputationResultType();

  QualType PromotionTypeLHS = getPromotionType(E->getComputationLHSType());
  QualType PromotionTypeRHS = getPromotionType(E->getRHS()->getType());

  if (!PromotionTypeRHS.isNull())
    OpInfo.RHS = CGF.emitPromotedScalarExpr(E->getRHS(), PromotionTypeRHS);
  else
    OpInfo.RHS = Visit(E->getRHS());

  OpInfo.FullType = PromotionTypeCR;
  OpInfo.CompType = OpInfo.FullType;
  if (auto VecType = dyn_cast_or_null<VectorType>(OpInfo.FullType)) {
    OpInfo.CompType = VecType->getElementType();
  }
  OpInfo.Opcode = E->getOpcode();
  OpInfo.FPFeatures = E->getFPFeaturesInEffect(CGF.getLangOpts());
  OpInfo.E = E;
  OpInfo.Loc = E->getSourceRange();

  // Load/convert the LHS
  LValue LHSLV = CGF.emitLValue(E->getLHS());

  if (const AtomicType *atomicTy = LHSTy->getAs<AtomicType>()) {
    assert(0 && "not implemented");
  }

  OpInfo.LHS = emitLoadOfLValue(LHSLV, E->getExprLoc());

  CIRGenFunction::SourceLocRAIIObject sourceloc{
      CGF, CGF.getLoc(E->getSourceRange())};
  SourceLocation Loc = E->getExprLoc();
  if (!PromotionTypeLHS.isNull())
    OpInfo.LHS = emitScalarConversion(OpInfo.LHS, LHSTy, PromotionTypeLHS,
                                      E->getExprLoc());
  else
    OpInfo.LHS = emitScalarConversion(OpInfo.LHS, LHSTy,
                                      E->getComputationLHSType(), Loc);

  // Expand the binary operator.
  Result = (this->*Func)(OpInfo);

  // Convert the result back to the LHS type,
  // potentially with Implicit Conversion sanitizer check.
  Result = emitScalarConversion(Result, PromotionTypeCR, LHSTy, Loc,
                                ScalarConversionOpts(CGF.SanOpts));

  // Store the result value into the LHS lvalue. Bit-fields are handled
  // specially because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after the
  // assignment...'.
  if (LHSLV.isBitField())
    CGF.emitStoreThroughBitfieldLValue(RValue::get(Result), LHSLV, Result);
  else
    CGF.emitStoreThroughLValue(RValue::get(Result), LHSLV);

  if (CGF.getLangOpts().OpenMP)
    CGF.CGM.getOpenMPRuntime().checkAndEmitLastprivateConditional(CGF,
                                                                  E->getLHS());
  return LHSLV;
}

mlir::Value ScalarExprEmitter::emitComplexToScalarConversion(mlir::Location Loc,
                                                             mlir::Value V,
                                                             CastKind Kind,
                                                             QualType DestTy) {
  cir::CastKind CastOpKind;
  switch (Kind) {
  case CK_FloatingComplexToReal:
    CastOpKind = cir::CastKind::float_complex_to_real;
    break;
  case CK_IntegralComplexToReal:
    CastOpKind = cir::CastKind::int_complex_to_real;
    break;
  case CK_FloatingComplexToBoolean:
    CastOpKind = cir::CastKind::float_complex_to_bool;
    break;
  case CK_IntegralComplexToBoolean:
    CastOpKind = cir::CastKind::int_complex_to_bool;
    break;
  default:
    llvm_unreachable("invalid complex-to-scalar cast kind");
  }

  return Builder.createCast(Loc, CastOpKind, V, CGF.convertType(DestTy));
}

mlir::Value ScalarExprEmitter::emitNullValue(QualType Ty, mlir::Location loc) {
  return CGF.emitFromMemory(CGF.CGM.emitNullConstant(Ty, loc), Ty);
}

mlir::Value ScalarExprEmitter::emitPromoted(const Expr *E,
                                            QualType PromotionType) {
  E = E->IgnoreParens();
  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {
#define HANDLE_BINOP(OP)                                                       \
  case BO_##OP:                                                                \
    return emit##OP(emitBinOps(BO, PromotionType));
      HANDLE_BINOP(Add)
      HANDLE_BINOP(Sub)
      HANDLE_BINOP(Mul)
      HANDLE_BINOP(Div)
#undef HANDLE_BINOP
    default:
      break;
    }
  } else if (const auto *UO = dyn_cast<UnaryOperator>(E)) {
    switch (UO->getOpcode()) {
    case UO_Imag:
    case UO_Real:
      llvm_unreachable("NYI");
    case UO_Minus:
      return VisitMinus(UO, PromotionType);
    case UO_Plus:
      return VisitPlus(UO, PromotionType);
    default:
      break;
    }
  }
  auto result = Visit(const_cast<Expr *>(E));
  if (result) {
    if (!PromotionType.isNull())
      return emitPromotedValue(result, PromotionType);
    return emitUnPromotedValue(result, E->getType());
  }
  return result;
}

mlir::Value ScalarExprEmitter::emitCompoundAssign(
    const CompoundAssignOperator *E,
    mlir::Value (ScalarExprEmitter::*Func)(const BinOpInfo &)) {

  bool Ignore = TestAndClearIgnoreResultAssign();
  mlir::Value RHS;
  LValue LHS = emitCompoundAssignLValue(E, Func, RHS);

  // If the result is clearly ignored, return now.
  if (Ignore)
    return {};

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return RHS;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LHS.isVolatileQualified())
    return RHS;

  // Otherwise, reload the value.
  return emitLoadOfLValue(LHS, E->getExprLoc());
}

mlir::Value ScalarExprEmitter::VisitExprWithCleanups(ExprWithCleanups *E) {
  auto scopeLoc = CGF.getLoc(E->getSourceRange());
  auto &builder = CGF.builder;

  auto scope = builder.create<cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Type &yieldTy, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                              builder.getInsertionBlock()};
        auto scopeYieldVal = Visit(E->getSubExpr());
        if (scopeYieldVal) {
          builder.create<cir::YieldOp>(loc, scopeYieldVal);
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
    LHS = emitCheckedLValue(E->getLHS(), CIRGenFunction::TCK_Store);

    // Store the value into the LHS. Bit-fields are handled specially because
    // the result is altered by the store, i.e., [C99 6.5.16p1]
    // 'An assignment expression has the value of the left operand after the
    // assignment...'.
    if (LHS.isBitField()) {
      CGF.emitStoreThroughBitfieldLValue(RValue::get(RHS), LHS, RHS);
    } else {
      CGF.emitNullabilityCheck(LHS, RHS, E->getExprLoc());
      CIRGenFunction::SourceLocRAIIObject loc{CGF,
                                              CGF.getLoc(E->getSourceRange())};
      CGF.emitStoreThroughLValue(RValue::get(RHS), LHS);
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
  return emitLoadOfLValue(LHS, E->getExprLoc());
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
        assert(!cir::MissingFeatures::incrementProfileCounter());
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
    assert(condExpr->getType()->isVectorType() && "?: op for SVE vector NYI");
    mlir::Value condValue = Visit(condExpr);
    mlir::Value lhsValue = Visit(lhsExpr);
    mlir::Value rhsValue = Visit(rhsExpr);
    return builder.create<cir::VecTernaryOp>(loc, condValue, lhsValue,
                                             rhsValue);
  }

  // If this is a really simple expression (like x ? 4 : 5), emit this as a
  // select instead of as control flow.  We can only do this if it is cheap and
  // safe to evaluate the LHS and RHS unconditionally.
  if (isCheapEnoughToEvaluateUnconditionally(lhsExpr, CGF) &&
      isCheapEnoughToEvaluateUnconditionally(rhsExpr, CGF)) {
    bool lhsIsVoid = false;
    auto condV = CGF.evaluateExprAsBool(condExpr);
    assert(!cir::MissingFeatures::incrementProfileCounter());

    return builder
        .create<cir::TernaryOp>(
            loc, condV, /*thenBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto lhs = Visit(lhsExpr);
              if (!lhs) {
                lhs = builder.getNullValue(CGF.VoidTy, loc);
                lhsIsVoid = true;
              }
              builder.create<cir::YieldOp>(loc, lhs);
            },
            /*elseBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto rhs = Visit(rhsExpr);
              if (lhsIsVoid) {
                assert(!rhs && "lhs and rhs types must match");
                rhs = builder.getNullValue(CGF.VoidTy, loc);
              }
              builder.create<cir::YieldOp>(loc, rhs);
            })
        .getResult();
  }

  mlir::Value condV = CGF.emitOpOnBoolExpr(loc, condExpr);
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
      if (mlir::isa<cir::VoidType>(yieldTy)) {
        builder.create<cir::YieldOp>(loc);
      } else { // Block returns: set null yield value.
        mlir::Value op0 = builder.getNullValue(yieldTy, loc);
        builder.create<cir::YieldOp>(loc, op0);
      }
    }
  };

  return builder
      .create<cir::TernaryOp>(
          loc, condV, /*trueBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                  b.getInsertionBlock()};
            CGF.currLexScope->setAsTernary();

            assert(!cir::MissingFeatures::incrementProfileCounter());
            eval.begin(CGF);
            auto lhs = Visit(lhsExpr);
            eval.end(CGF);

            if (lhs) {
              yieldTy = lhs.getType();
              b.create<cir::YieldOp>(loc, lhs);
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

            assert(!cir::MissingFeatures::incrementProfileCounter());
            eval.begin(CGF);
            auto rhs = Visit(rhsExpr);
            eval.end(CGF);

            if (rhs) {
              yieldTy = rhs.getType();
              b.create<cir::YieldOp>(loc, rhs);
            } else {
              // If LHS or RHS is a throw or void expression we need to patch
              // arms as to properly match yield types.
              insertPoints.push_back(b.saveInsertionPoint());
            }

            patchVoidOrThrowSites();
          })
      .getResult();
}

mlir::Value CIRGenFunction::emitScalarPrePostIncDec(const UnaryOperator *E,
                                                    LValue LV, bool isInc,
                                                    bool isPre) {
  return ScalarExprEmitter(*this, builder)
      .emitScalarPrePostIncDec(E, LV, isInc, isPre);
}

mlir::Value ScalarExprEmitter::VisitBinLAnd(const clang::BinaryOperator *E) {
  if (E->getType()->isVectorType()) {
    llvm_unreachable("NYI");
  }

  bool InstrumentRegions = CGF.CGM.getCodeGenOpts().hasProfileClangInstr();
  mlir::Type ResTy = convertType(E->getType());
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
      return Builder.getNullValue(ResTy, Loc);
  }

  CIRGenFunction::ConditionalEvaluation eval(CGF);

  mlir::Value LHSCondV = CGF.evaluateExprAsBool(E->getLHS());
  auto ResOp = Builder.create<cir::TernaryOp>(
      Loc, LHSCondV, /*trueBuilder=*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope LexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        mlir::Value RHSCondV = CGF.evaluateExprAsBool(E->getRHS());
        auto res = B.create<cir::TernaryOp>(
            Loc, RHSCondV, /*trueBuilder*/
            [&](mlir::OpBuilder &B, mlir::Location Loc) {
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    B.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = B.create<cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<cir::BoolAttr>(Builder.getBoolTy(), true));
              B.create<cir::YieldOp>(Loc, res.getRes());
            },
            /*falseBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location Loc) {
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    b.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<cir::BoolAttr>(Builder.getBoolTy(), false));
              b.create<cir::YieldOp>(Loc, res.getRes());
            });
        B.create<cir::YieldOp>(Loc, res.getResult());
      },
      /*falseBuilder*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        auto res = B.create<cir::ConstantOp>(
            Loc, Builder.getBoolTy(),
            Builder.getAttr<cir::BoolAttr>(Builder.getBoolTy(), false));
        B.create<cir::YieldOp>(Loc, res.getRes());
      });
  return Builder.createZExtOrBitCast(ResOp.getLoc(), ResOp.getResult(), ResTy);
}

mlir::Value ScalarExprEmitter::VisitBinLOr(const clang::BinaryOperator *E) {
  if (E->getType()->isVectorType()) {
    llvm_unreachable("NYI");
  }

  bool InstrumentRegions = CGF.CGM.getCodeGenOpts().hasProfileClangInstr();
  mlir::Type ResTy = convertType(E->getType());
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
    if (!CGF.ContainsLabel(E->getRHS())) {
      if (auto intTy = mlir::dyn_cast<cir::IntType>(ResTy))
        return Builder.getConstInt(Loc, intTy, 1);
      else
        return Builder.getBool(true, Loc);
    }
  }

  CIRGenFunction::ConditionalEvaluation eval(CGF);

  mlir::Value LHSCondV = CGF.evaluateExprAsBool(E->getLHS());
  auto ResOp = Builder.create<cir::TernaryOp>(
      Loc, LHSCondV, /*trueBuilder=*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        auto res = B.create<cir::ConstantOp>(
            Loc, Builder.getBoolTy(),
            Builder.getAttr<cir::BoolAttr>(Builder.getBoolTy(), true));
        B.create<cir::YieldOp>(Loc, res.getRes());
      },
      /*falseBuilder*/
      [&](mlir::OpBuilder &B, mlir::Location Loc) {
        CIRGenFunction::LexicalScope LexScope{CGF, Loc, B.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        mlir::Value RHSCondV = CGF.evaluateExprAsBool(E->getRHS());
        auto res = B.create<cir::TernaryOp>(
            Loc, RHSCondV, /*trueBuilder*/
            [&](mlir::OpBuilder &B, mlir::Location Loc) {
              SmallVector<mlir::Location, 2> Locs;
              if (mlir::isa<mlir::FileLineColLoc>(Loc)) {
                Locs.push_back(Loc);
                Locs.push_back(Loc);
              } else if (mlir::isa<mlir::FusedLoc>(Loc)) {
                auto fusedLoc = mlir::cast<mlir::FusedLoc>(Loc);
                Locs.push_back(fusedLoc.getLocations()[0]);
                Locs.push_back(fusedLoc.getLocations()[1]);
              }
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    B.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = B.create<cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<cir::BoolAttr>(Builder.getBoolTy(), true));
              B.create<cir::YieldOp>(Loc, res.getRes());
            },
            /*falseBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location Loc) {
              SmallVector<mlir::Location, 2> Locs;
              if (mlir::isa<mlir::FileLineColLoc>(Loc)) {
                Locs.push_back(Loc);
                Locs.push_back(Loc);
              } else if (mlir::isa<mlir::FusedLoc>(Loc)) {
                auto fusedLoc = mlir::cast<mlir::FusedLoc>(Loc);
                Locs.push_back(fusedLoc.getLocations()[0]);
                Locs.push_back(fusedLoc.getLocations()[1]);
              }
              CIRGenFunction::LexicalScope lexScope{CGF, Loc,
                                                    B.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<cir::ConstantOp>(
                  Loc, Builder.getBoolTy(),
                  Builder.getAttr<cir::BoolAttr>(Builder.getBoolTy(), false));
              b.create<cir::YieldOp>(Loc, res.getRes());
            });
        B.create<cir::YieldOp>(Loc, res.getResult());
      });

  return Builder.createZExtOrBitCast(ResOp.getLoc(), ResOp.getResult(), ResTy);
}

mlir::Value ScalarExprEmitter::VisitVAArgExpr(VAArgExpr *VE) {
  QualType Ty = VE->getType();

  if (Ty->isVariablyModifiedType())
    assert(!cir::MissingFeatures::variablyModifiedTypeEmission() && "NYI");

  Address ArgValue = Address::invalid();
  mlir::Value Val = CGF.emitVAArg(VE, ArgValue);

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

      if (E->isArgumentType()) {
        // sizeof(type) - make sure to emit the VLA size.
        CGF.emitVariablyModifiedType(TypeToSize);
      } else {
        // C99 6.5.3.4p2: If the argument is an expression of type
        // VLA, it is evaluated.
        CGF.emitIgnoredExpr(E->getArgumentExpr());
      }

      auto VlaSize = CGF.getVLASize(VAT);
      mlir::Value size = VlaSize.NumElts;

      // Scale the number of non-VLA elements by the non-VLA element size.
      CharUnits eltSize = CGF.getContext().getTypeSizeInChars(VlaSize.Type);
      if (!eltSize.isOne())
        size = Builder.createMul(size, CGF.CGM.getSize(eltSize).getValue());

      return size;
    }
  } else if (E->getKind() == UETT_OpenMPRequiredSimdAlign) {
    llvm_unreachable("NYI");
  }

  // If this isn't sizeof(vla), the result must be constant; use the constant
  // folding logic so we don't have to duplicate it here.
  return Builder.getConstInt(CGF.getLoc(E->getSourceRange()),
                             E->EvaluateKnownConstInt(CGF.getContext()));
}

mlir::Value CIRGenFunction::emitCheckedInBoundsGEP(
    mlir::Type ElemTy, mlir::Value Ptr, ArrayRef<mlir::Value> IdxList,
    bool SignedIndices, bool IsSubtraction, SourceLocation Loc) {
  mlir::Type PtrTy = Ptr.getType();
  assert(IdxList.size() == 1 && "multi-index ptr arithmetic NYI");
  mlir::Value GEPVal =
      builder.create<cir::PtrStrideOp>(CGM.getLoc(Loc), PtrTy, Ptr, IdxList[0]);

  // If the pointer overflow sanitizer isn't enabled, do nothing.
  if (!SanOpts.has(SanitizerKind::PointerOverflow))
    return GEPVal;

  // TODO(cir): the unreachable code below hides a substantial amount of code
  // from the original codegen related with pointer overflow sanitizer.
  assert(cir::MissingFeatures::pointerOverflowSanitizer());
  llvm_unreachable("pointer overflow sanitizer NYI");
}
