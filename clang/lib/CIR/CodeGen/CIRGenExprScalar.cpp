//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit Expr nodes with scalar CIR types as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenValue.h"

#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/MissingFeatures.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include <cassert>
#include <utility>

using namespace clang;
using namespace clang::CIRGen;

namespace {

struct BinOpInfo {
  mlir::Value lhs;
  mlir::Value rhs;
  SourceRange loc;
  QualType fullType;             // Type of operands and result
  QualType compType;             // Type used for computations. Element type
                                 // for vectors, otherwise same as FullType.
  BinaryOperator::Opcode opcode; // Opcode of BinOp to perform
  FPOptions fpfeatures;
  const Expr *e; // Entire expr, for error unsupported.  May not be binop.

  /// Check if the binop computes a division or a remainder.
  bool isDivRemOp() const {
    return opcode == BO_Div || opcode == BO_Rem || opcode == BO_DivAssign ||
           opcode == BO_RemAssign;
  }

  /// Check if the binop can result in integer overflow.
  bool mayHaveIntegerOverflow() const {
    // Without constant input, we can't rule out overflow.
    auto lhsci = lhs.getDefiningOp<cir::ConstantOp>();
    auto rhsci = rhs.getDefiningOp<cir::ConstantOp>();
    if (!lhsci || !rhsci)
      return true;

    assert(!cir::MissingFeatures::mayHaveIntegerOverflow());
    // TODO(cir): For now we just assume that we might overflow
    return true;
  }

  /// Check if at least one operand is a fixed point type. In such cases,
  /// this operation did not follow usual arithmetic conversion and both
  /// operands might not be of the same type.
  bool isFixedPointOp() const {
    // We cannot simply check the result type since comparison operations
    // return an int.
    if (const auto *binOp = llvm::dyn_cast<BinaryOperator>(e)) {
      QualType lhstype = binOp->getLHS()->getType();
      QualType rhstype = binOp->getRHS()->getType();
      return lhstype->isFixedPointType() || rhstype->isFixedPointType();
    }
    if (const auto *unop = llvm::dyn_cast<UnaryOperator>(e))
      return unop->getSubExpr()->getType()->isFixedPointType();
    return false;
  }
};

class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
  CIRGenFunction &cgf;
  CIRGenBuilderTy &builder;
  bool ignoreResultAssign;

public:
  ScalarExprEmitter(CIRGenFunction &cgf, CIRGenBuilderTy &builder)
      : cgf(cgf), builder(builder) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  mlir::Value emitComplexToScalarConversion(mlir::Location loc,
                                            mlir::Value value, CastKind kind,
                                            QualType destTy);

  mlir::Value emitNullValue(QualType ty, mlir::Location loc) {
    return cgf.cgm.emitNullConstant(ty, loc);
  }

  mlir::Value emitPromotedValue(mlir::Value result, QualType promotionType) {
    return builder.createFloatingCast(result, cgf.convertType(promotionType));
  }

  mlir::Value emitUnPromotedValue(mlir::Value result, QualType exprType) {
    return builder.createFloatingCast(result, cgf.convertType(exprType));
  }

  mlir::Value emitPromoted(const Expr *e, QualType promotionType);

  mlir::Value maybePromoteBoolResult(mlir::Value value,
                                     mlir::Type dstTy) const {
    if (mlir::isa<cir::IntType>(dstTy))
      return builder.createBoolToInt(value, dstTy);
    if (mlir::isa<cir::BoolType>(dstTy))
      return value;
    llvm_unreachable("Can only promote integer or boolean types");
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *e) {
    return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(e);
  }

  mlir::Value VisitStmt(Stmt *s) {
    llvm_unreachable("Statement passed to ScalarExprEmitter");
  }

  mlir::Value VisitExpr(Expr *e) {
    cgf.getCIRGenModule().errorNYI(
        e->getSourceRange(), "scalar expression kind: ", e->getStmtClassName());
    return {};
  }

  mlir::Value VisitPackIndexingExpr(PackIndexingExpr *e) {
    return Visit(e->getSelectedExpr());
  }

  mlir::Value VisitParenExpr(ParenExpr *pe) { return Visit(pe->getSubExpr()); }

  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *ge) {
    return Visit(ge->getResultExpr());
  }

  /// Emits the address of the l-value, then loads and returns the result.
  mlir::Value emitLoadOfLValue(const Expr *e) {
    LValue lv = cgf.emitLValue(e);
    // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
    return cgf.emitLoadOfLValue(lv, e->getExprLoc()).getValue();
  }

  mlir::Value emitLoadOfLValue(LValue lv, SourceLocation loc) {
    return cgf.emitLoadOfLValue(lv, loc).getValue();
  }

  // l-values
  mlir::Value VisitDeclRefExpr(DeclRefExpr *e) {
    if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(e))
      return cgf.emitScalarConstant(constant, e);

    return emitLoadOfLValue(e);
  }

  mlir::Value VisitIntegerLiteral(const IntegerLiteral *e) {
    mlir::Type type = cgf.convertType(e->getType());
    return builder.create<cir::ConstantOp>(
        cgf.getLoc(e->getExprLoc()), cir::IntAttr::get(type, e->getValue()));
  }

  mlir::Value VisitFloatingLiteral(const FloatingLiteral *e) {
    mlir::Type type = cgf.convertType(e->getType());
    assert(mlir::isa<cir::FPTypeInterface>(type) &&
           "expect floating-point type");
    return builder.create<cir::ConstantOp>(
        cgf.getLoc(e->getExprLoc()), cir::FPAttr::get(type, e->getValue()));
  }

  mlir::Value VisitCharacterLiteral(const CharacterLiteral *e) {
    mlir::Type ty = cgf.convertType(e->getType());
    auto init = cir::IntAttr::get(ty, e->getValue());
    return builder.create<cir::ConstantOp>(cgf.getLoc(e->getExprLoc()), init);
  }

  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *e) {
    return builder.getBool(e->getValue(), cgf.getLoc(e->getExprLoc()));
  }

  mlir::Value VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *e) {
    if (e->getType()->isVoidType())
      return {};

    return emitNullValue(e->getType(), cgf.getLoc(e->getSourceRange()));
  }

  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *e) {
    if (e->isGLValue())
      return emitLoadOfLValue(cgf.getOrCreateOpaqueLValueMapping(e),
                              e->getExprLoc());

    // Otherwise, assume the mapping is the scalar directly.
    return cgf.getOrCreateOpaqueRValueMapping(e).getValue();
  }

  mlir::Value VisitCastExpr(CastExpr *e);
  mlir::Value VisitCallExpr(const CallExpr *e);

  mlir::Value VisitStmtExpr(StmtExpr *e) {
    CIRGenFunction::StmtExprEvaluation eval(cgf);
    if (e->getType()->isVoidType()) {
      (void)cgf.emitCompoundStmt(*e->getSubStmt());
      return {};
    }

    Address retAlloca =
        cgf.createMemTemp(e->getType(), cgf.getLoc(e->getSourceRange()));
    (void)cgf.emitCompoundStmt(*e->getSubStmt(), &retAlloca);

    return cgf.emitLoadOfScalar(cgf.makeAddrLValue(retAlloca, e->getType()),
                                e->getExprLoc());
  }

  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *e) {
    if (e->getBase()->getType()->isVectorType()) {
      assert(!cir::MissingFeatures::scalableVectors());

      const mlir::Location loc = cgf.getLoc(e->getSourceRange());
      const mlir::Value vecValue = Visit(e->getBase());
      const mlir::Value indexValue = Visit(e->getIdx());
      return cgf.builder.create<cir::VecExtractOp>(loc, vecValue, indexValue);
    }
    // Just load the lvalue formed by the subscript expression.
    return emitLoadOfLValue(e);
  }

  mlir::Value VisitShuffleVectorExpr(ShuffleVectorExpr *e) {
    if (e->getNumSubExprs() == 2) {
      // The undocumented form of __builtin_shufflevector.
      mlir::Value inputVec = Visit(e->getExpr(0));
      mlir::Value indexVec = Visit(e->getExpr(1));
      return cgf.builder.create<cir::VecShuffleDynamicOp>(
          cgf.getLoc(e->getSourceRange()), inputVec, indexVec);
    }

    mlir::Value vec1 = Visit(e->getExpr(0));
    mlir::Value vec2 = Visit(e->getExpr(1));

    // The documented form of __builtin_shufflevector, where the indices are
    // a variable number of integer constants. The constants will be stored
    // in an ArrayAttr.
    SmallVector<mlir::Attribute, 8> indices;
    for (unsigned i = 2; i < e->getNumSubExprs(); ++i) {
      indices.push_back(
          cir::IntAttr::get(cgf.builder.getSInt64Ty(),
                            e->getExpr(i)
                                ->EvaluateKnownConstInt(cgf.getContext())
                                .getSExtValue()));
    }

    return cgf.builder.create<cir::VecShuffleOp>(
        cgf.getLoc(e->getSourceRange()), cgf.convertType(e->getType()), vec1,
        vec2, cgf.builder.getArrayAttr(indices));
  }

  mlir::Value VisitConvertVectorExpr(ConvertVectorExpr *e) {
    // __builtin_convertvector is an element-wise cast, and is implemented as a
    // regular cast. The back end handles casts of vectors correctly.
    return emitScalarConversion(Visit(e->getSrcExpr()),
                                e->getSrcExpr()->getType(), e->getType(),
                                e->getSourceRange().getBegin());
  }

  mlir::Value VisitMemberExpr(MemberExpr *e);

  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
    return emitLoadOfLValue(e);
  }

  mlir::Value VisitInitListExpr(InitListExpr *e);

  mlir::Value VisitExplicitCastExpr(ExplicitCastExpr *e) {
    return VisitCastExpr(e);
  }

  mlir::Value VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *e) {
    return cgf.cgm.emitNullConstant(e->getType(),
                                    cgf.getLoc(e->getSourceRange()));
  }

  /// Perform a pointer to boolean conversion.
  mlir::Value emitPointerToBoolConversion(mlir::Value v, QualType qt) {
    // TODO(cir): comparing the ptr to null is done when lowering CIR to LLVM.
    // We might want to have a separate pass for these types of conversions.
    return cgf.getBuilder().createPtrToBoolCast(v);
  }

  mlir::Value emitFloatToBoolConversion(mlir::Value src, mlir::Location loc) {
    cir::BoolType boolTy = builder.getBoolTy();
    return builder.create<cir::CastOp>(loc, boolTy,
                                       cir::CastKind::float_to_bool, src);
  }

  mlir::Value emitIntToBoolConversion(mlir::Value srcVal, mlir::Location loc) {
    // Because of the type rules of C, we often end up computing a
    // logical value, then zero extending it to int, then wanting it
    // as a logical value again.
    // TODO: optimize this common case here or leave it for later
    // CIR passes?
    cir::BoolType boolTy = builder.getBoolTy();
    return builder.create<cir::CastOp>(loc, boolTy, cir::CastKind::int_to_bool,
                                       srcVal);
  }

  /// Convert the specified expression value to a boolean (!cir.bool) truth
  /// value. This is equivalent to "Val != 0".
  mlir::Value emitConversionToBool(mlir::Value src, QualType srcType,
                                   mlir::Location loc) {
    assert(srcType.isCanonical() && "EmitScalarConversion strips typedefs");

    if (srcType->isRealFloatingType())
      return emitFloatToBoolConversion(src, loc);

    if (llvm::isa<MemberPointerType>(srcType)) {
      cgf.getCIRGenModule().errorNYI(loc, "member pointer to bool conversion");
      return builder.getFalse(loc);
    }

    if (srcType->isIntegerType())
      return emitIntToBoolConversion(src, loc);

    assert(::mlir::isa<cir::PointerType>(src.getType()));
    return emitPointerToBoolConversion(src, srcType);
  }

  // Emit a conversion from the specified type to the specified destination
  // type, both of which are CIR scalar types.
  struct ScalarConversionOpts {
    bool treatBooleanAsSigned;
    bool emitImplicitIntegerTruncationChecks;
    bool emitImplicitIntegerSignChangeChecks;

    ScalarConversionOpts()
        : treatBooleanAsSigned(false),
          emitImplicitIntegerTruncationChecks(false),
          emitImplicitIntegerSignChangeChecks(false) {}

    ScalarConversionOpts(clang::SanitizerSet sanOpts)
        : treatBooleanAsSigned(false),
          emitImplicitIntegerTruncationChecks(
              sanOpts.hasOneOf(SanitizerKind::ImplicitIntegerTruncation)),
          emitImplicitIntegerSignChangeChecks(
              sanOpts.has(SanitizerKind::ImplicitIntegerSignChange)) {}
  };

  // Conversion from bool, integral, or floating-point to integral or
  // floating-point. Conversions involving other types are handled elsewhere.
  // Conversion to bool is handled elsewhere because that's a comparison against
  // zero, not a simple cast. This handles both individual scalars and vectors.
  mlir::Value emitScalarCast(mlir::Value src, QualType srcType,
                             QualType dstType, mlir::Type srcTy,
                             mlir::Type dstTy, ScalarConversionOpts opts) {
    assert(!srcType->isMatrixType() && !dstType->isMatrixType() &&
           "Internal error: matrix types not handled by this function.");
    assert(!(mlir::isa<mlir::IntegerType>(srcTy) ||
             mlir::isa<mlir::IntegerType>(dstTy)) &&
           "Obsolete code. Don't use mlir::IntegerType with CIR.");

    mlir::Type fullDstTy = dstTy;
    if (mlir::isa<cir::VectorType>(srcTy) &&
        mlir::isa<cir::VectorType>(dstTy)) {
      // Use the element types of the vectors to figure out the CastKind.
      srcTy = mlir::dyn_cast<cir::VectorType>(srcTy).getElementType();
      dstTy = mlir::dyn_cast<cir::VectorType>(dstTy).getElementType();
    }

    std::optional<cir::CastKind> castKind;

    if (mlir::isa<cir::BoolType>(srcTy)) {
      if (opts.treatBooleanAsSigned)
        cgf.getCIRGenModule().errorNYI("signed bool");
      if (cgf.getBuilder().isInt(dstTy))
        castKind = cir::CastKind::bool_to_int;
      else if (mlir::isa<cir::FPTypeInterface>(dstTy))
        castKind = cir::CastKind::bool_to_float;
      else
        llvm_unreachable("Internal error: Cast to unexpected type");
    } else if (cgf.getBuilder().isInt(srcTy)) {
      if (cgf.getBuilder().isInt(dstTy))
        castKind = cir::CastKind::integral;
      else if (mlir::isa<cir::FPTypeInterface>(dstTy))
        castKind = cir::CastKind::int_to_float;
      else
        llvm_unreachable("Internal error: Cast to unexpected type");
    } else if (mlir::isa<cir::FPTypeInterface>(srcTy)) {
      if (cgf.getBuilder().isInt(dstTy)) {
        // If we can't recognize overflow as undefined behavior, assume that
        // overflow saturates. This protects against normal optimizations if we
        // are compiling with non-standard FP semantics.
        if (!cgf.cgm.getCodeGenOpts().StrictFloatCastOverflow)
          cgf.getCIRGenModule().errorNYI("strict float cast overflow");
        assert(!cir::MissingFeatures::fpConstraints());
        castKind = cir::CastKind::float_to_int;
      } else if (mlir::isa<cir::FPTypeInterface>(dstTy)) {
        // TODO: split this to createFPExt/createFPTrunc
        return builder.createFloatingCast(src, fullDstTy);
      } else {
        llvm_unreachable("Internal error: Cast to unexpected type");
      }
    } else {
      llvm_unreachable("Internal error: Cast from unexpected type");
    }

    assert(castKind.has_value() && "Internal error: CastKind not set.");
    return builder.create<cir::CastOp>(src.getLoc(), fullDstTy, *castKind, src);
  }

  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *e) {
    return Visit(e->getReplacement());
  }

  mlir::Value VisitVAArgExpr(VAArgExpr *ve) {
    QualType ty = ve->getType();

    if (ty->isVariablyModifiedType()) {
      cgf.cgm.errorNYI(ve->getSourceRange(),
                       "variably modified types in varargs");
    }

    return cgf.emitVAArg(ve);
  }

  mlir::Value VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *e);
  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *e);

  // Unary Operators.
  mlir::Value VisitUnaryPostDec(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, cir::UnaryOpKind::Dec, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, cir::UnaryOpKind::Inc, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, cir::UnaryOpKind::Dec, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, cir::UnaryOpKind::Inc, true);
  }
  mlir::Value emitScalarPrePostIncDec(const UnaryOperator *e, LValue lv,
                                      cir::UnaryOpKind kind, bool isPre) {
    if (cgf.getLangOpts().OpenMP)
      cgf.cgm.errorNYI(e->getSourceRange(), "inc/dec OpenMP");

    QualType type = e->getSubExpr()->getType();

    mlir::Value value;
    mlir::Value input;

    if (type->getAs<AtomicType>()) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Atomic inc/dec");
      // TODO(cir): This is not correct, but it will produce reasonable code
      // until atomic operations are implemented.
      value = cgf.emitLoadOfLValue(lv, e->getExprLoc()).getValue();
      input = value;
    } else {
      value = cgf.emitLoadOfLValue(lv, e->getExprLoc()).getValue();
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
    if (kind == cir::UnaryOpKind::Inc && type->isBooleanType()) {
      value = builder.getTrue(cgf.getLoc(e->getExprLoc()));
    } else if (type->isIntegerType()) {
      QualType promotedType;
      [[maybe_unused]] bool canPerformLossyDemotionCheck = false;
      if (cgf.getContext().isPromotableIntegerType(type)) {
        promotedType = cgf.getContext().getPromotedIntegerType(type);
        assert(promotedType != type && "Shouldn't promote to the same type.");
        canPerformLossyDemotionCheck = true;
        canPerformLossyDemotionCheck &=
            cgf.getContext().getCanonicalType(type) !=
            cgf.getContext().getCanonicalType(promotedType);
        canPerformLossyDemotionCheck &=
            type->isIntegerType() && promotedType->isIntegerType();

        // TODO(cir): Currently, we store bitwidths in CIR types only for
        // integers. This might also be required for other types.

        assert(
            (!canPerformLossyDemotionCheck ||
             type->isSignedIntegerOrEnumerationType() ||
             promotedType->isSignedIntegerOrEnumerationType() ||
             mlir::cast<cir::IntType>(cgf.convertType(type)).getWidth() ==
                 mlir::cast<cir::IntType>(cgf.convertType(type)).getWidth()) &&
            "The following check expects that if we do promotion to different "
            "underlying canonical type, at least one of the types (either "
            "base or promoted) will be signed, or the bitwidths will match.");
      }

      assert(!cir::MissingFeatures::sanitizers());
      if (e->canOverflow() && type->isSignedIntegerOrEnumerationType()) {
        value = emitIncDecConsiderOverflowBehavior(e, value, kind);
      } else {
        cir::UnaryOpKind kind =
            e->isIncrementOp() ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec;
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        value = emitUnaryOp(e, kind, input, /*nsw=*/false);
      }
    } else if (const PointerType *ptr = type->getAs<PointerType>()) {
      QualType type = ptr->getPointeeType();
      if (cgf.getContext().getAsVariableArrayType(type)) {
        // VLA types don't have constant size.
        cgf.cgm.errorNYI(e->getSourceRange(), "Pointer arithmetic on VLA");
        return {};
      } else if (type->isFunctionType()) {
        // Arithmetic on function pointers (!) is just +-1.
        cgf.cgm.errorNYI(e->getSourceRange(),
                         "Pointer arithmetic on function pointer");
        return {};
      } else {
        // For everything else, we can just do a simple increment.
        mlir::Location loc = cgf.getLoc(e->getSourceRange());
        CIRGenBuilderTy &builder = cgf.getBuilder();
        int amount = kind == cir::UnaryOpKind::Inc ? 1 : -1;
        mlir::Value amt = builder.getSInt32(amount, loc);
        assert(!cir::MissingFeatures::sanitizers());
        value = builder.createPtrStride(loc, value, amt);
      }
    } else if (type->isVectorType()) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec vector");
      return {};
    } else if (type->isRealFloatingType()) {
      assert(!cir::MissingFeatures::cgFPOptionsRAII());

      if (type->isHalfType() &&
          !cgf.getContext().getLangOpts().NativeHalfType) {
        cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec half");
        return {};
      }

      if (mlir::isa<cir::SingleType, cir::DoubleType>(value.getType())) {
        // Create the inc/dec operation.
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        assert(kind == cir::UnaryOpKind::Inc ||
               kind == cir::UnaryOpKind::Dec && "Invalid UnaryOp kind");
        value = emitUnaryOp(e, kind, value);
      } else {
        cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec other fp type");
        return {};
      }
    } else if (type->isFixedPointType()) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec other fixed point");
      return {};
    } else {
      assert(type->castAs<ObjCObjectPointerType>());
      cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec ObjectiveC pointer");
      return {};
    }

    CIRGenFunction::SourceLocRAIIObject sourceloc{
        cgf, cgf.getLoc(e->getSourceRange())};

    // Store the updated result through the lvalue
    if (lv.isBitField())
      return cgf.emitStoreThroughBitfieldLValue(RValue::get(value), lv);
    else
      cgf.emitStoreThroughLValue(RValue::get(value), lv);

    // If this is a postinc, return the value read from memory, otherwise use
    // the updated value.
    return isPre ? value : input;
  }

  mlir::Value emitIncDecConsiderOverflowBehavior(const UnaryOperator *e,
                                                 mlir::Value inVal,
                                                 cir::UnaryOpKind kind) {
    assert(kind == cir::UnaryOpKind::Inc ||
           kind == cir::UnaryOpKind::Dec && "Invalid UnaryOp kind");
    switch (cgf.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      return emitUnaryOp(e, kind, inVal, /*nsw=*/false);
    case LangOptions::SOB_Undefined:
      assert(!cir::MissingFeatures::sanitizers());
      return emitUnaryOp(e, kind, inVal, /*nsw=*/true);
    case LangOptions::SOB_Trapping:
      if (!e->canOverflow())
        return emitUnaryOp(e, kind, inVal, /*nsw=*/true);
      cgf.cgm.errorNYI(e->getSourceRange(), "inc/def overflow SOB_Trapping");
      return {};
    }
    llvm_unreachable("Unexpected signed overflow behavior kind");
  }

  mlir::Value VisitUnaryAddrOf(const UnaryOperator *e) {
    if (llvm::isa<MemberPointerType>(e->getType())) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Address of member pointer");
      return builder.getNullPtr(cgf.convertType(e->getType()),
                                cgf.getLoc(e->getExprLoc()));
    }

    return cgf.emitLValue(e->getSubExpr()).getPointer();
  }

  mlir::Value VisitUnaryDeref(const UnaryOperator *e) {
    if (e->getType()->isVoidType())
      return Visit(e->getSubExpr()); // the actual value should be unused
    return emitLoadOfLValue(e);
  }

  mlir::Value VisitUnaryPlus(const UnaryOperator *e) {
    QualType promotionType = getPromotionType(e->getSubExpr()->getType());
    mlir::Value result =
        emitUnaryPlusOrMinus(e, cir::UnaryOpKind::Plus, promotionType);
    if (result && !promotionType.isNull())
      return emitUnPromotedValue(result, e->getType());
    return result;
  }

  mlir::Value VisitUnaryMinus(const UnaryOperator *e) {
    QualType promotionType = getPromotionType(e->getSubExpr()->getType());
    mlir::Value result =
        emitUnaryPlusOrMinus(e, cir::UnaryOpKind::Minus, promotionType);
    if (result && !promotionType.isNull())
      return emitUnPromotedValue(result, e->getType());
    return result;
  }

  mlir::Value emitUnaryPlusOrMinus(const UnaryOperator *e,
                                   cir::UnaryOpKind kind,
                                   QualType promotionType) {
    ignoreResultAssign = false;
    mlir::Value operand;
    if (!promotionType.isNull())
      operand = cgf.emitPromotedScalarExpr(e->getSubExpr(), promotionType);
    else
      operand = Visit(e->getSubExpr());

    bool nsw =
        kind == cir::UnaryOpKind::Minus && e->getType()->isSignedIntegerType();

    // NOTE: LLVM codegen will lower this directly to either a FNeg
    // or a Sub instruction.  In CIR this will be handled later in LowerToLLVM.
    return emitUnaryOp(e, kind, operand, nsw);
  }

  mlir::Value emitUnaryOp(const UnaryOperator *e, cir::UnaryOpKind kind,
                          mlir::Value input, bool nsw = false) {
    return builder.create<cir::UnaryOp>(
        cgf.getLoc(e->getSourceRange().getBegin()), input.getType(), kind,
        input, nsw);
  }

  mlir::Value VisitUnaryNot(const UnaryOperator *e) {
    ignoreResultAssign = false;
    mlir::Value op = Visit(e->getSubExpr());
    return emitUnaryOp(e, cir::UnaryOpKind::Not, op);
  }

  mlir::Value VisitUnaryLNot(const UnaryOperator *e);

  mlir::Value VisitUnaryReal(const UnaryOperator *e);
  mlir::Value VisitUnaryImag(const UnaryOperator *e);
  mlir::Value VisitRealImag(const UnaryOperator *e,
                            QualType promotionType = QualType());

  mlir::Value VisitUnaryExtension(const UnaryOperator *e) {
    return Visit(e->getSubExpr());
  }

  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    CIRGenFunction::CXXDefaultInitExprScope scope(cgf, die);
    return Visit(die->getExpr());
  }

  mlir::Value VisitCXXThisExpr(CXXThisExpr *te) { return cgf.loadCXXThis(); }

  mlir::Value VisitExprWithCleanups(ExprWithCleanups *e);
  mlir::Value VisitCXXNewExpr(const CXXNewExpr *e) {
    return cgf.emitCXXNewExpr(e);
  }
  mlir::Value VisitCXXDeleteExpr(const CXXDeleteExpr *e) {
    cgf.emitCXXDeleteExpr(e);
    return {};
  }

  mlir::Value VisitCXXThrowExpr(const CXXThrowExpr *e) {
    cgf.emitCXXThrowExpr(e);
    return {};
  }

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  /// TODO: do we need ScalarConversionOpts here? Should be done in another
  /// pass.
  mlir::Value
  emitScalarConversion(mlir::Value src, QualType srcType, QualType dstType,
                       SourceLocation loc,
                       ScalarConversionOpts opts = ScalarConversionOpts()) {
    // All conversions involving fixed point types should be handled by the
    // emitFixedPoint family functions. This is done to prevent bloating up
    // this function more, and although fixed point numbers are represented by
    // integers, we do not want to follow any logic that assumes they should be
    // treated as integers.
    // TODO(leonardchan): When necessary, add another if statement checking for
    // conversions to fixed point types from other types.
    // conversions to fixed point types from other types.
    if (srcType->isFixedPointType() || dstType->isFixedPointType()) {
      cgf.getCIRGenModule().errorNYI(loc, "fixed point conversions");
      return {};
    }

    srcType = srcType.getCanonicalType();
    dstType = dstType.getCanonicalType();
    if (srcType == dstType) {
      if (opts.emitImplicitIntegerSignChangeChecks)
        cgf.getCIRGenModule().errorNYI(loc,
                                       "implicit integer sign change checks");
      return src;
    }

    if (dstType->isVoidType())
      return {};

    mlir::Type mlirSrcType = src.getType();

    // Handle conversions to bool first, they are special: comparisons against
    // 0.
    if (dstType->isBooleanType())
      return emitConversionToBool(src, srcType, cgf.getLoc(loc));

    mlir::Type mlirDstType = cgf.convertType(dstType);

    if (srcType->isHalfType() &&
        !cgf.getContext().getLangOpts().NativeHalfType) {
      // Cast to FP using the intrinsic if the half type itself isn't supported.
      if (mlir::isa<cir::FPTypeInterface>(mlirDstType)) {
        if (cgf.getContext().getTargetInfo().useFP16ConversionIntrinsics())
          cgf.getCIRGenModule().errorNYI(loc,
                                         "cast via llvm.convert.from.fp16");
      } else {
        // Cast to other types through float, using either the intrinsic or
        // FPExt, depending on whether the half type itself is supported (as
        // opposed to operations on half, available with NativeHalfType).
        if (cgf.getContext().getTargetInfo().useFP16ConversionIntrinsics())
          cgf.getCIRGenModule().errorNYI(loc,
                                         "cast via llvm.convert.from.fp16");
        // FIXME(cir): For now lets pretend we shouldn't use the conversion
        // intrinsics and insert a cast here unconditionally.
        src = builder.createCast(cgf.getLoc(loc), cir::CastKind::floating, src,
                                 cgf.FloatTy);
        srcType = cgf.getContext().FloatTy;
        mlirSrcType = cgf.FloatTy;
      }
    }

    // TODO(cir): LLVM codegen ignore conversions like int -> uint,
    // is there anything to be done for CIR here?
    if (mlirSrcType == mlirDstType) {
      if (opts.emitImplicitIntegerSignChangeChecks)
        cgf.getCIRGenModule().errorNYI(loc,
                                       "implicit integer sign change checks");
      return src;
    }

    // Handle pointer conversions next: pointers can only be converted to/from
    // other pointers and integers. Check for pointer types in terms of LLVM, as
    // some native types (like Obj-C id) may map to a pointer type.
    if (auto dstPT = dyn_cast<cir::PointerType>(mlirDstType)) {
      cgf.getCIRGenModule().errorNYI(loc, "pointer casts");
      return builder.getNullPtr(dstPT, src.getLoc());
    }

    if (isa<cir::PointerType>(mlirSrcType)) {
      // Must be an ptr to int cast.
      assert(isa<cir::IntType>(mlirDstType) && "not ptr->int?");
      return builder.createPtrToInt(src, mlirDstType);
    }

    // A scalar can be splatted to an extended vector of the same element type
    if (dstType->isExtVectorType() && !srcType->isVectorType()) {
      // Sema should add casts to make sure that the source expression's type
      // is the same as the vector's element type (sans qualifiers)
      assert(dstType->castAs<ExtVectorType>()->getElementType().getTypePtr() ==
                 srcType.getTypePtr() &&
             "Splatted expr doesn't match with vector element type?");

      cgf.getCIRGenModule().errorNYI(loc, "vector splatting");
      return {};
    }

    if (srcType->isMatrixType() && dstType->isMatrixType()) {
      cgf.getCIRGenModule().errorNYI(loc,
                                     "matrix type to matrix type conversion");
      return {};
    }
    assert(!srcType->isMatrixType() && !dstType->isMatrixType() &&
           "Internal error: conversion between matrix type and scalar type");

    // Finally, we have the arithmetic types or vectors of arithmetic types.
    mlir::Value res = nullptr;
    mlir::Type resTy = mlirDstType;

    res = emitScalarCast(src, srcType, dstType, mlirSrcType, mlirDstType, opts);

    if (mlirDstType != resTy) {
      if (cgf.getContext().getTargetInfo().useFP16ConversionIntrinsics()) {
        cgf.getCIRGenModule().errorNYI(loc, "cast via llvm.convert.to.fp16");
      }
      // FIXME(cir): For now we never use FP16 conversion intrinsics even if
      // required by the target. Change that once this is implemented
      res = builder.createCast(cgf.getLoc(loc), cir::CastKind::floating, res,
                               resTy);
    }

    if (opts.emitImplicitIntegerTruncationChecks)
      cgf.getCIRGenModule().errorNYI(loc, "implicit integer truncation checks");

    if (opts.emitImplicitIntegerSignChangeChecks)
      cgf.getCIRGenModule().errorNYI(loc,
                                     "implicit integer sign change checks");

    return res;
  }

  BinOpInfo emitBinOps(const BinaryOperator *e,
                       QualType promotionType = QualType()) {
    BinOpInfo result;
    result.lhs = cgf.emitPromotedScalarExpr(e->getLHS(), promotionType);
    result.rhs = cgf.emitPromotedScalarExpr(e->getRHS(), promotionType);
    if (!promotionType.isNull())
      result.fullType = promotionType;
    else
      result.fullType = e->getType();
    result.compType = result.fullType;
    if (const auto *vecType = dyn_cast_or_null<VectorType>(result.fullType)) {
      result.compType = vecType->getElementType();
    }
    result.opcode = e->getOpcode();
    result.loc = e->getSourceRange();
    // TODO(cir): Result.FPFeatures
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    result.e = e;
    return result;
  }

  mlir::Value emitMul(const BinOpInfo &ops);
  mlir::Value emitDiv(const BinOpInfo &ops);
  mlir::Value emitRem(const BinOpInfo &ops);
  mlir::Value emitAdd(const BinOpInfo &ops);
  mlir::Value emitSub(const BinOpInfo &ops);
  mlir::Value emitShl(const BinOpInfo &ops);
  mlir::Value emitShr(const BinOpInfo &ops);
  mlir::Value emitAnd(const BinOpInfo &ops);
  mlir::Value emitXor(const BinOpInfo &ops);
  mlir::Value emitOr(const BinOpInfo &ops);

  LValue emitCompoundAssignLValue(
      const CompoundAssignOperator *e,
      mlir::Value (ScalarExprEmitter::*f)(const BinOpInfo &),
      mlir::Value &result);
  mlir::Value
  emitCompoundAssign(const CompoundAssignOperator *e,
                     mlir::Value (ScalarExprEmitter::*f)(const BinOpInfo &));

  // TODO(cir): Candidate to be in a common AST helper between CIR and LLVM
  // codegen.
  QualType getPromotionType(QualType ty) {
    const clang::ASTContext &ctx = cgf.getContext();
    if (auto *complexTy = ty->getAs<ComplexType>()) {
      QualType elementTy = complexTy->getElementType();
      if (elementTy.UseExcessPrecision(ctx))
        return ctx.getComplexType(ctx.FloatTy);
    }

    if (ty.UseExcessPrecision(cgf.getContext())) {
      if (auto *vt = ty->getAs<VectorType>()) {
        unsigned numElements = vt->getNumElements();
        return ctx.getVectorType(ctx.FloatTy, numElements, vt->getVectorKind());
      }
      return cgf.getContext().FloatTy;
    }

    return QualType();
  }

// Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *e) {                          \
    QualType promotionTy = getPromotionType(e->getType());                     \
    auto result = emit##OP(emitBinOps(e, promotionTy));                        \
    if (result && !promotionTy.isNull())                                       \
      result = emitUnPromotedValue(result, e->getType());                      \
    return result;                                                             \
  }                                                                            \
  mlir::Value VisitBin##OP##Assign(const CompoundAssignOperator *e) {          \
    return emitCompoundAssign(e, &ScalarExprEmitter::emit##OP);                \
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

  mlir::Value emitCmp(const BinaryOperator *e) {
    const mlir::Location loc = cgf.getLoc(e->getExprLoc());
    mlir::Value result;
    QualType lhsTy = e->getLHS()->getType();
    QualType rhsTy = e->getRHS()->getType();

    auto clangCmpToCIRCmp =
        [](clang::BinaryOperatorKind clangCmp) -> cir::CmpOpKind {
      switch (clangCmp) {
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
        llvm_unreachable("unsupported comparison kind for cir.cmp");
      }
    };

    cir::CmpOpKind kind = clangCmpToCIRCmp(e->getOpcode());
    if (lhsTy->getAs<MemberPointerType>()) {
      assert(!cir::MissingFeatures::dataMemberType());
      assert(e->getOpcode() == BO_EQ || e->getOpcode() == BO_NE);
      mlir::Value lhs = cgf.emitScalarExpr(e->getLHS());
      mlir::Value rhs = cgf.emitScalarExpr(e->getRHS());
      result = builder.createCompare(loc, kind, lhs, rhs);
    } else if (!lhsTy->isAnyComplexType() && !rhsTy->isAnyComplexType()) {
      BinOpInfo boInfo = emitBinOps(e);
      mlir::Value lhs = boInfo.lhs;
      mlir::Value rhs = boInfo.rhs;

      if (lhsTy->isVectorType()) {
        if (!e->getType()->isVectorType()) {
          // If AltiVec, the comparison results in a numeric type, so we use
          // intrinsics comparing vectors and giving 0 or 1 as a result
          cgf.cgm.errorNYI(loc, "AltiVec comparison");
        } else {
          // Other kinds of vectors. Element-wise comparison returning
          // a vector.
          result = builder.create<cir::VecCmpOp>(
              cgf.getLoc(boInfo.loc), cgf.convertType(boInfo.fullType), kind,
              boInfo.lhs, boInfo.rhs);
        }
      } else if (boInfo.isFixedPointOp()) {
        assert(!cir::MissingFeatures::fixedPointType());
        cgf.cgm.errorNYI(loc, "fixed point comparisons");
        result = builder.getBool(false, loc);
      } else {
        // integers and pointers
        if (cgf.cgm.getCodeGenOpts().StrictVTablePointers &&
            mlir::isa<cir::PointerType>(lhs.getType()) &&
            mlir::isa<cir::PointerType>(rhs.getType())) {
          cgf.cgm.errorNYI(loc, "strict vtable pointer comparisons");
        }

        cir::CmpOpKind kind = clangCmpToCIRCmp(e->getOpcode());
        result = builder.createCompare(loc, kind, lhs, rhs);
      }
    } else {
      // Complex Comparison: can only be an equality comparison.
      assert(e->getOpcode() == BO_EQ || e->getOpcode() == BO_NE);

      BinOpInfo boInfo = emitBinOps(e);
      result = builder.create<cir::CmpOp>(loc, kind, boInfo.lhs, boInfo.rhs);
    }

    return emitScalarConversion(result, cgf.getContext().BoolTy, e->getType(),
                                e->getExprLoc());
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

  mlir::Value VisitBinAssign(const BinaryOperator *e) {
    const bool ignore = std::exchange(ignoreResultAssign, false);

    mlir::Value rhs;
    LValue lhs;

    switch (e->getLHS()->getType().getObjCLifetime()) {
    case Qualifiers::OCL_Strong:
    case Qualifiers::OCL_Autoreleasing:
    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Weak:
      assert(!cir::MissingFeatures::objCLifetime());
      break;
    case Qualifiers::OCL_None:
      // __block variables need to have the rhs evaluated first, plus this
      // should improve codegen just a little.
      rhs = Visit(e->getRHS());
      assert(!cir::MissingFeatures::sanitizers());
      // TODO(cir): This needs to be emitCheckedLValue() once we support
      // sanitizers
      lhs = cgf.emitLValue(e->getLHS());

      // Store the value into the LHS. Bit-fields are handled specially because
      // the result is altered by the store, i.e., [C99 6.5.16p1]
      // 'An assignment expression has the value of the left operand after the
      // assignment...'.
      if (lhs.isBitField()) {
        rhs = cgf.emitStoreThroughBitfieldLValue(RValue::get(rhs), lhs);
      } else {
        cgf.emitNullabilityCheck(lhs, rhs, e->getExprLoc());
        CIRGenFunction::SourceLocRAIIObject loc{
            cgf, cgf.getLoc(e->getSourceRange())};
        cgf.emitStoreThroughLValue(RValue::get(rhs), lhs);
      }
    }

    // If the result is clearly ignored, return now.
    if (ignore)
      return nullptr;

    // The result of an assignment in C is the assigned r-value.
    if (!cgf.getLangOpts().CPlusPlus)
      return rhs;

    // If the lvalue is non-volatile, return the computed value of the
    // assignment.
    if (!lhs.isVolatile())
      return rhs;

    // Otherwise, reload the value.
    return emitLoadOfLValue(lhs, e->getExprLoc());
  }

  mlir::Value VisitBinComma(const BinaryOperator *e) {
    cgf.emitIgnoredExpr(e->getLHS());
    // NOTE: We don't need to EnsureInsertPoint() like LLVM codegen.
    return Visit(e->getRHS());
  }

  mlir::Value VisitBinLAnd(const clang::BinaryOperator *e) {
    if (e->getType()->isVectorType()) {
      mlir::Location loc = cgf.getLoc(e->getExprLoc());
      auto vecTy = mlir::cast<cir::VectorType>(cgf.convertType(e->getType()));
      mlir::Value zeroValue = builder.getNullValue(vecTy.getElementType(), loc);
      SmallVector<mlir::Value, 16> elements(vecTy.getSize(), zeroValue);
      auto zeroVec = cir::VecCreateOp::create(builder, loc, vecTy, elements);

      mlir::Value lhs = Visit(e->getLHS());
      mlir::Value rhs = Visit(e->getRHS());

      auto cmpOpKind = cir::CmpOpKind::ne;
      lhs = cir::VecCmpOp::create(builder, loc, vecTy, cmpOpKind, lhs, zeroVec);
      rhs = cir::VecCmpOp::create(builder, loc, vecTy, cmpOpKind, rhs, zeroVec);
      mlir::Value vecOr = builder.createAnd(loc, lhs, rhs);
      return builder.createIntCast(vecOr, vecTy);
    }

    assert(!cir::MissingFeatures::instrumentation());
    mlir::Type resTy = cgf.convertType(e->getType());
    mlir::Location loc = cgf.getLoc(e->getExprLoc());

    CIRGenFunction::ConditionalEvaluation eval(cgf);

    mlir::Value lhsCondV = cgf.evaluateExprAsBool(e->getLHS());
    auto resOp = builder.create<cir::TernaryOp>(
        loc, lhsCondV, /*trueBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          CIRGenFunction::LexicalScope lexScope{cgf, loc,
                                                b.getInsertionBlock()};
          cgf.curLexScope->setAsTernary();
          mlir::Value res = cgf.evaluateExprAsBool(e->getRHS());
          lexScope.forceCleanup();
          cir::YieldOp::create(b, loc, res);
        },
        /*falseBuilder*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          CIRGenFunction::LexicalScope lexScope{cgf, loc,
                                                b.getInsertionBlock()};
          cgf.curLexScope->setAsTernary();
          auto res = cir::ConstantOp::create(b, loc, builder.getFalseAttr());
          cir::YieldOp::create(b, loc, res.getRes());
        });
    return maybePromoteBoolResult(resOp.getResult(), resTy);
  }

  mlir::Value VisitBinLOr(const clang::BinaryOperator *e) {
    if (e->getType()->isVectorType()) {
      mlir::Location loc = cgf.getLoc(e->getExprLoc());
      auto vecTy = mlir::cast<cir::VectorType>(cgf.convertType(e->getType()));
      mlir::Value zeroValue = builder.getNullValue(vecTy.getElementType(), loc);
      SmallVector<mlir::Value, 16> elements(vecTy.getSize(), zeroValue);
      auto zeroVec = cir::VecCreateOp::create(builder, loc, vecTy, elements);

      mlir::Value lhs = Visit(e->getLHS());
      mlir::Value rhs = Visit(e->getRHS());

      auto cmpOpKind = cir::CmpOpKind::ne;
      lhs = cir::VecCmpOp::create(builder, loc, vecTy, cmpOpKind, lhs, zeroVec);
      rhs = cir::VecCmpOp::create(builder, loc, vecTy, cmpOpKind, rhs, zeroVec);
      mlir::Value vecOr = builder.createOr(loc, lhs, rhs);
      return builder.createIntCast(vecOr, vecTy);
    }

    assert(!cir::MissingFeatures::instrumentation());
    mlir::Type resTy = cgf.convertType(e->getType());
    mlir::Location loc = cgf.getLoc(e->getExprLoc());

    CIRGenFunction::ConditionalEvaluation eval(cgf);

    mlir::Value lhsCondV = cgf.evaluateExprAsBool(e->getLHS());
    auto resOp = builder.create<cir::TernaryOp>(
        loc, lhsCondV, /*trueBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          CIRGenFunction::LexicalScope lexScope{cgf, loc,
                                                b.getInsertionBlock()};
          cgf.curLexScope->setAsTernary();
          auto res = cir::ConstantOp::create(b, loc, builder.getTrueAttr());
          cir::YieldOp::create(b, loc, res.getRes());
        },
        /*falseBuilder*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          CIRGenFunction::LexicalScope lexScope{cgf, loc,
                                                b.getInsertionBlock()};
          cgf.curLexScope->setAsTernary();
          mlir::Value res = cgf.evaluateExprAsBool(e->getRHS());
          lexScope.forceCleanup();
          cir::YieldOp::create(b, loc, res);
        });

    return maybePromoteBoolResult(resOp.getResult(), resTy);
  }

  mlir::Value VisitAtomicExpr(AtomicExpr *e) {
    return cgf.emitAtomicExpr(e).getValue();
  }
};

LValue ScalarExprEmitter::emitCompoundAssignLValue(
    const CompoundAssignOperator *e,
    mlir::Value (ScalarExprEmitter::*func)(const BinOpInfo &),
    mlir::Value &result) {
  if (e->getComputationResultType()->isAnyComplexType())
    return cgf.emitScalarCompoundAssignWithComplex(e, result);

  QualType lhsTy = e->getLHS()->getType();
  BinOpInfo opInfo;

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.

  QualType promotionTypeCR = getPromotionType(e->getComputationResultType());
  if (promotionTypeCR.isNull())
    promotionTypeCR = e->getComputationResultType();

  QualType promotionTypeLHS = getPromotionType(e->getComputationLHSType());
  QualType promotionTypeRHS = getPromotionType(e->getRHS()->getType());

  if (!promotionTypeRHS.isNull())
    opInfo.rhs = cgf.emitPromotedScalarExpr(e->getRHS(), promotionTypeRHS);
  else
    opInfo.rhs = Visit(e->getRHS());

  opInfo.fullType = promotionTypeCR;
  opInfo.compType = opInfo.fullType;
  if (const auto *vecType = dyn_cast_or_null<VectorType>(opInfo.fullType))
    opInfo.compType = vecType->getElementType();
  opInfo.opcode = e->getOpcode();
  opInfo.fpfeatures = e->getFPFeaturesInEffect(cgf.getLangOpts());
  opInfo.e = e;
  opInfo.loc = e->getSourceRange();

  // Load/convert the LHS
  LValue lhsLV = cgf.emitLValue(e->getLHS());

  if (lhsTy->getAs<AtomicType>()) {
    cgf.cgm.errorNYI(result.getLoc(), "atomic lvalue assign");
    return LValue();
  }

  opInfo.lhs = emitLoadOfLValue(lhsLV, e->getExprLoc());

  CIRGenFunction::SourceLocRAIIObject sourceloc{
      cgf, cgf.getLoc(e->getSourceRange())};
  SourceLocation loc = e->getExprLoc();
  if (!promotionTypeLHS.isNull())
    opInfo.lhs = emitScalarConversion(opInfo.lhs, lhsTy, promotionTypeLHS, loc);
  else
    opInfo.lhs = emitScalarConversion(opInfo.lhs, lhsTy,
                                      e->getComputationLHSType(), loc);

  // Expand the binary operator.
  result = (this->*func)(opInfo);

  // Convert the result back to the LHS type,
  // potentially with Implicit Conversion sanitizer check.
  result = emitScalarConversion(result, promotionTypeCR, lhsTy, loc,
                                ScalarConversionOpts(cgf.sanOpts));

  // Store the result value into the LHS lvalue. Bit-fields are handled
  // specially because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after the
  // assignment...'.
  if (lhsLV.isBitField())
    cgf.emitStoreThroughBitfieldLValue(RValue::get(result), lhsLV);
  else
    cgf.emitStoreThroughLValue(RValue::get(result), lhsLV);

  if (cgf.getLangOpts().OpenMP)
    cgf.cgm.errorNYI(e->getSourceRange(), "openmp");

  return lhsLV;
}

mlir::Value ScalarExprEmitter::emitComplexToScalarConversion(mlir::Location lov,
                                                             mlir::Value value,
                                                             CastKind kind,
                                                             QualType destTy) {
  cir::CastKind castOpKind;
  switch (kind) {
  case CK_FloatingComplexToReal:
    castOpKind = cir::CastKind::float_complex_to_real;
    break;
  case CK_IntegralComplexToReal:
    castOpKind = cir::CastKind::int_complex_to_real;
    break;
  case CK_FloatingComplexToBoolean:
    castOpKind = cir::CastKind::float_complex_to_bool;
    break;
  case CK_IntegralComplexToBoolean:
    castOpKind = cir::CastKind::int_complex_to_bool;
    break;
  default:
    llvm_unreachable("invalid complex-to-scalar cast kind");
  }

  return builder.createCast(lov, castOpKind, value, cgf.convertType(destTy));
}

mlir::Value ScalarExprEmitter::emitPromoted(const Expr *e,
                                            QualType promotionType) {
  e = e->IgnoreParens();
  if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
    switch (bo->getOpcode()) {
#define HANDLE_BINOP(OP)                                                       \
  case BO_##OP:                                                                \
    return emit##OP(emitBinOps(bo, promotionType));
      HANDLE_BINOP(Add)
      HANDLE_BINOP(Sub)
      HANDLE_BINOP(Mul)
      HANDLE_BINOP(Div)
#undef HANDLE_BINOP
    default:
      break;
    }
  } else if (const auto *uo = dyn_cast<UnaryOperator>(e)) {
    switch (uo->getOpcode()) {
    case UO_Imag:
    case UO_Real:
      return VisitRealImag(uo, promotionType);
    case UO_Minus:
      return emitUnaryPlusOrMinus(uo, cir::UnaryOpKind::Minus, promotionType);
    case UO_Plus:
      return emitUnaryPlusOrMinus(uo, cir::UnaryOpKind::Plus, promotionType);
    default:
      break;
    }
  }
  mlir::Value result = Visit(const_cast<Expr *>(e));
  if (result) {
    if (!promotionType.isNull())
      return emitPromotedValue(result, promotionType);
    return emitUnPromotedValue(result, e->getType());
  }
  return result;
}

mlir::Value ScalarExprEmitter::emitCompoundAssign(
    const CompoundAssignOperator *e,
    mlir::Value (ScalarExprEmitter::*func)(const BinOpInfo &)) {

  bool ignore = std::exchange(ignoreResultAssign, false);
  mlir::Value rhs;
  LValue lhs = emitCompoundAssignLValue(e, func, rhs);

  // If the result is clearly ignored, return now.
  if (ignore)
    return {};

  // The result of an assignment in C is the assigned r-value.
  if (!cgf.getLangOpts().CPlusPlus)
    return rhs;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!lhs.isVolatile())
    return rhs;

  // Otherwise, reload the value.
  return emitLoadOfLValue(lhs, e->getExprLoc());
}

mlir::Value ScalarExprEmitter::VisitExprWithCleanups(ExprWithCleanups *e) {
  mlir::Location scopeLoc = cgf.getLoc(e->getSourceRange());
  mlir::OpBuilder &builder = cgf.builder;

  auto scope = cir::ScopeOp::create(
      builder, scopeLoc,
      /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Type &yieldTy, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{cgf, loc,
                                              builder.getInsertionBlock()};
        mlir::Value scopeYieldVal = Visit(e->getSubExpr());
        if (scopeYieldVal) {
          // Defend against dominance problems caused by jumps out of expression
          // evaluation through the shared cleanup block.
          lexScope.forceCleanup();
          cir::YieldOp::create(builder, loc, scopeYieldVal);
          yieldTy = scopeYieldVal.getType();
        }
      });

  return scope.getNumResults() > 0 ? scope->getResult(0) : nullptr;
}

} // namespace

LValue
CIRGenFunction::emitCompoundAssignmentLValue(const CompoundAssignOperator *e) {
  ScalarExprEmitter emitter(*this, builder);
  mlir::Value result;
  switch (e->getOpcode()) {
#define COMPOUND_OP(Op)                                                        \
  case BO_##Op##Assign:                                                        \
    return emitter.emitCompoundAssignLValue(e, &ScalarExprEmitter::emit##Op,   \
                                            result)
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

/// Emit the computation of the specified expression of scalar type.
mlir::Value CIRGenFunction::emitScalarExpr(const Expr *e) {
  assert(e && hasScalarEvaluationKind(e->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(e));
}

mlir::Value CIRGenFunction::emitPromotedScalarExpr(const Expr *e,
                                                   QualType promotionType) {
  if (!promotionType.isNull())
    return ScalarExprEmitter(*this, builder).emitPromoted(e, promotionType);
  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(e));
}

[[maybe_unused]] static bool mustVisitNullValue(const Expr *e) {
  // If a null pointer expression's type is the C++0x nullptr_t and
  // the expression is not a simple literal, it must be evaluated
  // for its potential side effects.
  if (isa<IntegerLiteral>(e) || isa<CXXNullPtrLiteralExpr>(e))
    return false;
  return e->getType()->isNullPtrType();
}

/// If \p e is a widened promoted integer, get its base (unpromoted) type.
static std::optional<QualType>
getUnwidenedIntegerType(const ASTContext &astContext, const Expr *e) {
  const Expr *base = e->IgnoreImpCasts();
  if (e == base)
    return std::nullopt;

  QualType baseTy = base->getType();
  if (!astContext.isPromotableIntegerType(baseTy) ||
      astContext.getTypeSize(baseTy) >= astContext.getTypeSize(e->getType()))
    return std::nullopt;

  return baseTy;
}

/// Check if \p e is a widened promoted integer.
[[maybe_unused]] static bool isWidenedIntegerOp(const ASTContext &astContext,
                                                const Expr *e) {
  return getUnwidenedIntegerType(astContext, e).has_value();
}

/// Check if we can skip the overflow check for \p Op.
[[maybe_unused]] static bool canElideOverflowCheck(const ASTContext &astContext,
                                                   const BinOpInfo &op) {
  assert((isa<UnaryOperator>(op.e) || isa<BinaryOperator>(op.e)) &&
         "Expected a unary or binary operator");

  // If the binop has constant inputs and we can prove there is no overflow,
  // we can elide the overflow check.
  if (!op.mayHaveIntegerOverflow())
    return true;

  // If a unary op has a widened operand, the op cannot overflow.
  if (const auto *uo = dyn_cast<UnaryOperator>(op.e))
    return !uo->canOverflow();

  // We usually don't need overflow checks for binops with widened operands.
  // Multiplication with promoted unsigned operands is a special case.
  const auto *bo = cast<BinaryOperator>(op.e);
  std::optional<QualType> optionalLHSTy =
      getUnwidenedIntegerType(astContext, bo->getLHS());
  if (!optionalLHSTy)
    return false;

  std::optional<QualType> optionalRHSTy =
      getUnwidenedIntegerType(astContext, bo->getRHS());
  if (!optionalRHSTy)
    return false;

  QualType lhsTy = *optionalLHSTy;
  QualType rhsTy = *optionalRHSTy;

  // This is the simple case: binops without unsigned multiplication, and with
  // widened operands. No overflow check is needed here.
  if ((op.opcode != BO_Mul && op.opcode != BO_MulAssign) ||
      !lhsTy->isUnsignedIntegerType() || !rhsTy->isUnsignedIntegerType())
    return true;

  // For unsigned multiplication the overflow check can be elided if either one
  // of the unpromoted types are less than half the size of the promoted type.
  unsigned promotedSize = astContext.getTypeSize(op.e->getType());
  return (2 * astContext.getTypeSize(lhsTy)) < promotedSize ||
         (2 * astContext.getTypeSize(rhsTy)) < promotedSize;
}

/// Emit pointer + index arithmetic.
static mlir::Value emitPointerArithmetic(CIRGenFunction &cgf,
                                         const BinOpInfo &op,
                                         bool isSubtraction) {
  // Must have binary (not unary) expr here.  Unary pointer
  // increment/decrement doesn't use this path.
  const BinaryOperator *expr = cast<BinaryOperator>(op.e);

  mlir::Value pointer = op.lhs;
  Expr *pointerOperand = expr->getLHS();
  mlir::Value index = op.rhs;
  Expr *indexOperand = expr->getRHS();

  // In the case of subtraction, the FE has ensured that the LHS is always the
  // pointer. However, addition can have the pointer on either side. We will
  // always have a pointer operand and an integer operand, so if the LHS wasn't
  // a pointer, we need to swap our values.
  if (!isSubtraction && !mlir::isa<cir::PointerType>(pointer.getType())) {
    std::swap(pointer, index);
    std::swap(pointerOperand, indexOperand);
  }
  assert(mlir::isa<cir::PointerType>(pointer.getType()) &&
         "Need a pointer operand");
  assert(mlir::isa<cir::IntType>(index.getType()) && "Need an integer operand");

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
          cgf.getContext(), op.opcode, expr->getLHS(), expr->getRHS()))
    return cgf.getBuilder().createIntToPtr(index, pointer.getType());

  // Differently from LLVM codegen, ABI bits for index sizes is handled during
  // LLVM lowering.

  // If this is subtraction, negate the index.
  if (isSubtraction)
    index = cgf.getBuilder().createNeg(index);

  assert(!cir::MissingFeatures::sanitizers());

  const PointerType *pointerType =
      pointerOperand->getType()->getAs<PointerType>();
  if (!pointerType) {
    cgf.cgm.errorNYI("Objective-C:pointer arithmetic with non-pointer type");
    return nullptr;
  }

  QualType elementType = pointerType->getPointeeType();
  if (cgf.getContext().getAsVariableArrayType(elementType)) {
    cgf.cgm.errorNYI("variable array type");
    return nullptr;
  }

  if (elementType->isVoidType() || elementType->isFunctionType()) {
    cgf.cgm.errorNYI("void* or function pointer arithmetic");
    return nullptr;
  }

  assert(!cir::MissingFeatures::sanitizers());
  return cgf.getBuilder().create<cir::PtrStrideOp>(
      cgf.getLoc(op.e->getExprLoc()), pointer.getType(), pointer, index);
}

mlir::Value ScalarExprEmitter::emitMul(const BinOpInfo &ops) {
  const mlir::Location loc = cgf.getLoc(ops.loc);
  if (ops.compType->isSignedIntegerOrEnumerationType()) {
    switch (cgf.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      if (!cgf.sanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return builder.createMul(loc, ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Undefined:
      if (!cgf.sanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return builder.createNSWMul(loc, ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      if (canElideOverflowCheck(cgf.getContext(), ops))
        return builder.createNSWMul(loc, ops.lhs, ops.rhs);
      cgf.cgm.errorNYI("sanitizers");
    }
  }
  if (ops.fullType->isConstantMatrixType()) {
    assert(!cir::MissingFeatures::matrixType());
    cgf.cgm.errorNYI("matrix types");
    return nullptr;
  }
  if (ops.compType->isUnsignedIntegerType() &&
      cgf.sanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
      !canElideOverflowCheck(cgf.getContext(), ops))
    cgf.cgm.errorNYI("unsigned int overflow sanitizer");

  if (cir::isFPOrVectorOfFPType(ops.lhs.getType())) {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    return builder.createFMul(loc, ops.lhs, ops.rhs);
  }

  if (ops.isFixedPointOp()) {
    assert(!cir::MissingFeatures::fixedPointType());
    cgf.cgm.errorNYI("fixed point");
    return nullptr;
  }

  return builder.create<cir::BinOp>(cgf.getLoc(ops.loc),
                                    cgf.convertType(ops.fullType),
                                    cir::BinOpKind::Mul, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::emitDiv(const BinOpInfo &ops) {
  return builder.create<cir::BinOp>(cgf.getLoc(ops.loc),
                                    cgf.convertType(ops.fullType),
                                    cir::BinOpKind::Div, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::emitRem(const BinOpInfo &ops) {
  return builder.create<cir::BinOp>(cgf.getLoc(ops.loc),
                                    cgf.convertType(ops.fullType),
                                    cir::BinOpKind::Rem, ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::emitAdd(const BinOpInfo &ops) {
  if (mlir::isa<cir::PointerType>(ops.lhs.getType()) ||
      mlir::isa<cir::PointerType>(ops.rhs.getType()))
    return emitPointerArithmetic(cgf, ops, /*isSubtraction=*/false);

  const mlir::Location loc = cgf.getLoc(ops.loc);
  if (ops.compType->isSignedIntegerOrEnumerationType()) {
    switch (cgf.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      if (!cgf.sanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return builder.createAdd(loc, ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Undefined:
      if (!cgf.sanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return builder.createNSWAdd(loc, ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      if (canElideOverflowCheck(cgf.getContext(), ops))
        return builder.createNSWAdd(loc, ops.lhs, ops.rhs);
      cgf.cgm.errorNYI("sanitizers");
    }
  }
  if (ops.fullType->isConstantMatrixType()) {
    assert(!cir::MissingFeatures::matrixType());
    cgf.cgm.errorNYI("matrix types");
    return nullptr;
  }

  if (ops.compType->isUnsignedIntegerType() &&
      cgf.sanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
      !canElideOverflowCheck(cgf.getContext(), ops))
    cgf.cgm.errorNYI("unsigned int overflow sanitizer");

  if (cir::isFPOrVectorOfFPType(ops.lhs.getType())) {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    return builder.createFAdd(loc, ops.lhs, ops.rhs);
  }

  if (ops.isFixedPointOp()) {
    assert(!cir::MissingFeatures::fixedPointType());
    cgf.cgm.errorNYI("fixed point");
    return {};
  }

  return builder.create<cir::BinOp>(loc, cgf.convertType(ops.fullType),
                                    cir::BinOpKind::Add, ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::emitSub(const BinOpInfo &ops) {
  const mlir::Location loc = cgf.getLoc(ops.loc);
  // The LHS is always a pointer if either side is.
  if (!mlir::isa<cir::PointerType>(ops.lhs.getType())) {
    if (ops.compType->isSignedIntegerOrEnumerationType()) {
      switch (cgf.getLangOpts().getSignedOverflowBehavior()) {
      case LangOptions::SOB_Defined: {
        if (!cgf.sanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return builder.createSub(loc, ops.lhs, ops.rhs);
        [[fallthrough]];
      }
      case LangOptions::SOB_Undefined:
        if (!cgf.sanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return builder.createNSWSub(loc, ops.lhs, ops.rhs);
        [[fallthrough]];
      case LangOptions::SOB_Trapping:
        if (canElideOverflowCheck(cgf.getContext(), ops))
          return builder.createNSWSub(loc, ops.lhs, ops.rhs);
        cgf.cgm.errorNYI("sanitizers");
      }
    }

    if (ops.fullType->isConstantMatrixType()) {
      assert(!cir::MissingFeatures::matrixType());
      cgf.cgm.errorNYI("matrix types");
      return nullptr;
    }

    if (ops.compType->isUnsignedIntegerType() &&
        cgf.sanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
        !canElideOverflowCheck(cgf.getContext(), ops))
      cgf.cgm.errorNYI("unsigned int overflow sanitizer");

    if (cir::isFPOrVectorOfFPType(ops.lhs.getType())) {
      assert(!cir::MissingFeatures::cgFPOptionsRAII());
      return builder.createFSub(loc, ops.lhs, ops.rhs);
    }

    if (ops.isFixedPointOp()) {
      assert(!cir::MissingFeatures::fixedPointType());
      cgf.cgm.errorNYI("fixed point");
      return {};
    }

    return builder.create<cir::BinOp>(cgf.getLoc(ops.loc),
                                      cgf.convertType(ops.fullType),
                                      cir::BinOpKind::Sub, ops.lhs, ops.rhs);
  }

  // If the RHS is not a pointer, then we have normal pointer
  // arithmetic.
  if (!mlir::isa<cir::PointerType>(ops.rhs.getType()))
    return emitPointerArithmetic(cgf, ops, /*isSubtraction=*/true);

  // Otherwise, this is a pointer subtraction

  // Do the raw subtraction part.
  //
  // TODO(cir): note for LLVM lowering out of this; when expanding this into
  // LLVM we shall take VLA's, division by element size, etc.
  //
  // See more in `EmitSub` in CGExprScalar.cpp.
  assert(!cir::MissingFeatures::ptrDiffOp());
  cgf.cgm.errorNYI("ptrdiff");
  return {};
}

mlir::Value ScalarExprEmitter::emitShl(const BinOpInfo &ops) {
  // TODO: This misses out on the sanitizer check below.
  if (ops.isFixedPointOp()) {
    assert(cir::MissingFeatures::fixedPointType());
    cgf.cgm.errorNYI("fixed point");
    return {};
  }

  // CIR accepts shift between different types, meaning nothing special
  // to be done here. OTOH, LLVM requires the LHS and RHS to be the same type:
  // promote or truncate the RHS to the same size as the LHS.

  bool sanitizeSignedBase = cgf.sanOpts.has(SanitizerKind::ShiftBase) &&
                            ops.compType->hasSignedIntegerRepresentation() &&
                            !cgf.getLangOpts().isSignedOverflowDefined() &&
                            !cgf.getLangOpts().CPlusPlus20;
  bool sanitizeUnsignedBase =
      cgf.sanOpts.has(SanitizerKind::UnsignedShiftBase) &&
      ops.compType->hasUnsignedIntegerRepresentation();
  bool sanitizeBase = sanitizeSignedBase || sanitizeUnsignedBase;
  bool sanitizeExponent = cgf.sanOpts.has(SanitizerKind::ShiftExponent);

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (cgf.getLangOpts().OpenCL)
    cgf.cgm.errorNYI("opencl");
  else if ((sanitizeBase || sanitizeExponent) &&
           mlir::isa<cir::IntType>(ops.lhs.getType()))
    cgf.cgm.errorNYI("sanitizers");

  return builder.createShiftLeft(cgf.getLoc(ops.loc), ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::emitShr(const BinOpInfo &ops) {
  // TODO: This misses out on the sanitizer check below.
  if (ops.isFixedPointOp()) {
    assert(cir::MissingFeatures::fixedPointType());
    cgf.cgm.errorNYI("fixed point");
    return {};
  }

  // CIR accepts shift between different types, meaning nothing special
  // to be done here. OTOH, LLVM requires the LHS and RHS to be the same type:
  // promote or truncate the RHS to the same size as the LHS.

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (cgf.getLangOpts().OpenCL)
    cgf.cgm.errorNYI("opencl");
  else if (cgf.sanOpts.has(SanitizerKind::ShiftExponent) &&
           mlir::isa<cir::IntType>(ops.lhs.getType()))
    cgf.cgm.errorNYI("sanitizers");

  // Note that we don't need to distinguish unsigned treatment at this
  // point since it will be handled later by LLVM lowering.
  return builder.createShiftRight(cgf.getLoc(ops.loc), ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::emitAnd(const BinOpInfo &ops) {
  return builder.create<cir::BinOp>(cgf.getLoc(ops.loc),
                                    cgf.convertType(ops.fullType),
                                    cir::BinOpKind::And, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::emitXor(const BinOpInfo &ops) {
  return builder.create<cir::BinOp>(cgf.getLoc(ops.loc),
                                    cgf.convertType(ops.fullType),
                                    cir::BinOpKind::Xor, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::emitOr(const BinOpInfo &ops) {
  return builder.create<cir::BinOp>(cgf.getLoc(ops.loc),
                                    cgf.convertType(ops.fullType),
                                    cir::BinOpKind::Or, ops.lhs, ops.rhs);
}

// Emit code for an explicit or implicit cast.  Implicit
// casts have to handle a more broad range of conversions than explicit
// casts, as they handle things like function to ptr-to-function decay
// etc.
mlir::Value ScalarExprEmitter::VisitCastExpr(CastExpr *ce) {
  Expr *subExpr = ce->getSubExpr();
  QualType destTy = ce->getType();
  CastKind kind = ce->getCastKind();

  // These cases are generally not written to ignore the result of evaluating
  // their sub-expressions, so we clear this now.
  ignoreResultAssign = false;

  switch (kind) {
  case clang::CK_Dependent:
    llvm_unreachable("dependent cast kind in CIR gen!");
  case clang::CK_BuiltinFnToFnPtr:
    llvm_unreachable("builtin functions are handled elsewhere");

  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_BitCast: {
    mlir::Value src = Visit(const_cast<Expr *>(subExpr));
    mlir::Type dstTy = cgf.convertType(destTy);

    assert(!cir::MissingFeatures::addressSpace());

    if (cgf.sanOpts.has(SanitizerKind::CFIUnrelatedCast))
      cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(),
                                     "sanitizer support");

    if (cgf.cgm.getCodeGenOpts().StrictVTablePointers)
      cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(),
                                     "strict vtable pointers");

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

    return cgf.getBuilder().createBitcast(cgf.getLoc(subExpr->getSourceRange()),
                                          src, dstTy);
  }

  case CK_AtomicToNonAtomic: {
    cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(),
                                   "CastExpr: ", ce->getCastKindName());
    mlir::Location loc = cgf.getLoc(subExpr->getSourceRange());
    return cgf.createDummyValue(loc, destTy);
  }
  case CK_NonAtomicToAtomic:
  case CK_UserDefinedConversion:
    return Visit(const_cast<Expr *>(subExpr));
  case CK_NoOp: {
    auto v = Visit(const_cast<Expr *>(subExpr));
    if (v) {
      // CK_NoOp can model a pointer qualification conversion, which can remove
      // an array bound and change the IR type.
      // FIXME: Once pointee types are removed from IR, remove this.
      mlir::Type t = cgf.convertType(destTy);
      if (t != v.getType())
        cgf.getCIRGenModule().errorNYI("pointer qualification conversion");
    }
    return v;
  }
  case CK_IntegralToPointer: {
    mlir::Type destCIRTy = cgf.convertType(destTy);
    mlir::Value src = Visit(const_cast<Expr *>(subExpr));

    // Properly resize by casting to an int of the same size as the pointer.
    // Clang's IntegralToPointer includes 'bool' as the source, but in CIR
    // 'bool' is not an integral type.  So check the source type to get the
    // correct CIR conversion.
    mlir::Type middleTy = cgf.cgm.getDataLayout().getIntPtrType(destCIRTy);
    mlir::Value middleVal = builder.createCast(
        subExpr->getType()->isBooleanType() ? cir::CastKind::bool_to_int
                                            : cir::CastKind::integral,
        src, middleTy);

    if (cgf.cgm.getCodeGenOpts().StrictVTablePointers) {
      cgf.cgm.errorNYI(subExpr->getSourceRange(),
                       "IntegralToPointer: strict vtable pointers");
      return {};
    }

    return builder.createIntToPtr(middleVal, destCIRTy);
  }

  case CK_ArrayToPointerDecay:
    return cgf.emitArrayToPointerDecay(subExpr).getPointer();

  case CK_NullToPointer: {
    if (mustVisitNullValue(subExpr))
      cgf.emitIgnoredExpr(subExpr);

    // Note that DestTy is used as the MLIR type instead of a custom
    // nullptr type.
    mlir::Type ty = cgf.convertType(destTy);
    return builder.getNullPtr(ty, cgf.getLoc(subExpr->getExprLoc()));
  }

  case CK_LValueToRValue:
    assert(cgf.getContext().hasSameUnqualifiedType(subExpr->getType(), destTy));
    assert(subExpr->isGLValue() && "lvalue-to-rvalue applied to r-value!");
    return Visit(const_cast<Expr *>(subExpr));

  case CK_IntegralCast: {
    ScalarConversionOpts opts;
    if (auto *ice = dyn_cast<ImplicitCastExpr>(ce)) {
      if (!ice->isPartOfExplicitCast())
        opts = ScalarConversionOpts(cgf.sanOpts);
    }
    return emitScalarConversion(Visit(subExpr), subExpr->getType(), destTy,
                                ce->getExprLoc(), opts);
  }

  case CK_FloatingComplexToReal:
  case CK_IntegralComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToBoolean: {
    mlir::Value value = cgf.emitComplexExpr(subExpr);
    return emitComplexToScalarConversion(cgf.getLoc(ce->getExprLoc()), value,
                                         kind, destTy);
  }

  case CK_FloatingRealToComplex:
  case CK_FloatingComplexCast:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_FloatingComplexToIntegralComplex:
    llvm_unreachable("scalar cast to non-scalar value");

  case CK_PointerToIntegral: {
    assert(!destTy->isBooleanType() && "bool should use PointerToBool");
    if (cgf.cgm.getCodeGenOpts().StrictVTablePointers)
      cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(),
                                     "strict vtable pointers");
    return builder.createPtrToInt(Visit(subExpr), cgf.convertType(destTy));
  }
  case CK_ToVoid:
    cgf.emitIgnoredExpr(subExpr);
    return {};

  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingCast:
  case CK_FixedPointToFloating:
  case CK_FloatingToFixedPoint: {
    if (kind == CK_FixedPointToFloating || kind == CK_FloatingToFixedPoint) {
      cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(),
                                     "fixed point casts");
      return {};
    }
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    return emitScalarConversion(Visit(subExpr), subExpr->getType(), destTy,
                                ce->getExprLoc());
  }

  case CK_IntegralToBoolean:
    return emitIntToBoolConversion(Visit(subExpr),
                                   cgf.getLoc(ce->getSourceRange()));

  case CK_PointerToBoolean:
    return emitPointerToBoolConversion(Visit(subExpr), subExpr->getType());
  case CK_FloatingToBoolean:
    return emitFloatToBoolConversion(Visit(subExpr),
                                     cgf.getLoc(subExpr->getExprLoc()));
  case CK_MemberPointerToBoolean: {
    mlir::Value memPtr = Visit(subExpr);
    return builder.createCast(cgf.getLoc(ce->getSourceRange()),
                              cir::CastKind::member_ptr_to_bool, memPtr,
                              cgf.convertType(destTy));
  }

  case CK_VectorSplat: {
    // Create a vector object and fill all elements with the same scalar value.
    assert(destTy->isVectorType() && "CK_VectorSplat to non-vector type");
    return builder.create<cir::VecSplatOp>(
        cgf.getLoc(subExpr->getSourceRange()), cgf.convertType(destTy),
        Visit(subExpr));
  }
  case CK_FunctionToPointerDecay:
    return cgf.emitLValue(subExpr).getPointer();

  default:
    cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(),
                                   "CastExpr: ", ce->getCastKindName());
  }
  return {};
}

mlir::Value ScalarExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(cgf.getContext())->isReferenceType())
    return emitLoadOfLValue(e);

  auto v = cgf.emitCallExpr(e).getValue();
  assert(!cir::MissingFeatures::emitLValueAlignmentAssumption());
  return v;
}

mlir::Value ScalarExprEmitter::VisitMemberExpr(MemberExpr *e) {
  // TODO(cir): The classic codegen calls tryEmitAsConstant() here. Folding
  // constants sound like work for MLIR optimizers, but we'll keep an assertion
  // for now.
  assert(!cir::MissingFeatures::tryEmitAsConstant());
  Expr::EvalResult result;
  if (e->EvaluateAsInt(result, cgf.getContext(), Expr::SE_AllowSideEffects)) {
    cgf.cgm.errorNYI(e->getSourceRange(), "Constant interger member expr");
    // Fall through to emit this as a non-constant access.
  }
  return emitLoadOfLValue(e);
}

mlir::Value ScalarExprEmitter::VisitInitListExpr(InitListExpr *e) {
  const unsigned numInitElements = e->getNumInits();

  if (e->hadArrayRangeDesignator()) {
    cgf.cgm.errorNYI(e->getSourceRange(), "ArrayRangeDesignator");
    return {};
  }

  if (e->getType()->isVectorType()) {
    const auto vectorType =
        mlir::cast<cir::VectorType>(cgf.convertType(e->getType()));

    SmallVector<mlir::Value, 16> elements;
    for (Expr *init : e->inits()) {
      elements.push_back(Visit(init));
    }

    // Zero-initialize any remaining values.
    if (numInitElements < vectorType.getSize()) {
      const mlir::Value zeroValue = cgf.getBuilder().getNullValue(
          vectorType.getElementType(), cgf.getLoc(e->getSourceRange()));
      std::fill_n(std::back_inserter(elements),
                  vectorType.getSize() - numInitElements, zeroValue);
    }

    return cgf.getBuilder().create<cir::VecCreateOp>(
        cgf.getLoc(e->getSourceRange()), vectorType, elements);
  }

  // C++11 value-initialization for the scalar.
  if (numInitElements == 0)
    return emitNullValue(e->getType(), cgf.getLoc(e->getExprLoc()));

  return Visit(e->getInit(0));
}

mlir::Value CIRGenFunction::emitScalarConversion(mlir::Value src,
                                                 QualType srcTy, QualType dstTy,
                                                 SourceLocation loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(srcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(dstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this, builder)
      .emitScalarConversion(src, srcTy, dstTy, loc);
}

mlir::Value CIRGenFunction::emitComplexToScalarConversion(mlir::Value src,
                                                          QualType srcTy,
                                                          QualType dstTy,
                                                          SourceLocation loc) {
  assert(srcTy->isAnyComplexType() && hasScalarEvaluationKind(dstTy) &&
         "Invalid complex -> scalar conversion");

  QualType complexElemTy = srcTy->castAs<ComplexType>()->getElementType();
  if (dstTy->isBooleanType()) {
    auto kind = complexElemTy->isFloatingType()
                    ? cir::CastKind::float_complex_to_bool
                    : cir::CastKind::int_complex_to_bool;
    return builder.createCast(getLoc(loc), kind, src, convertType(dstTy));
  }

  auto kind = complexElemTy->isFloatingType()
                  ? cir::CastKind::float_complex_to_real
                  : cir::CastKind::int_complex_to_real;
  mlir::Value real =
      builder.createCast(getLoc(loc), kind, src, convertType(complexElemTy));
  return emitScalarConversion(real, complexElemTy, dstTy, loc);
}

mlir::Value ScalarExprEmitter::VisitUnaryLNot(const UnaryOperator *e) {
  // Perform vector logical not on comparison with zero vector.
  if (e->getType()->isVectorType() &&
      e->getType()->castAs<VectorType>()->getVectorKind() ==
          VectorKind::Generic) {
    mlir::Value oper = Visit(e->getSubExpr());
    mlir::Location loc = cgf.getLoc(e->getExprLoc());
    auto operVecTy = mlir::cast<cir::VectorType>(oper.getType());
    auto exprVecTy = mlir::cast<cir::VectorType>(cgf.convertType(e->getType()));
    mlir::Value zeroVec = builder.getNullValue(operVecTy, loc);
    return cir::VecCmpOp::create(builder, loc, exprVecTy, cir::CmpOpKind::eq,
                                 oper, zeroVec);
  }

  // Compare operand to zero.
  mlir::Value boolVal = cgf.evaluateExprAsBool(e->getSubExpr());

  // Invert value.
  boolVal = builder.createNot(boolVal);

  // ZExt result to the expr type.
  return maybePromoteBoolResult(boolVal, cgf.convertType(e->getType()));
}

mlir::Value ScalarExprEmitter::VisitUnaryReal(const UnaryOperator *e) {
  QualType promotionTy = getPromotionType(e->getSubExpr()->getType());
  mlir::Value result = VisitRealImag(e, promotionTy);
  if (result && !promotionTy.isNull())
    result = emitUnPromotedValue(result, e->getType());
  return result;
}

mlir::Value ScalarExprEmitter::VisitUnaryImag(const UnaryOperator *e) {
  QualType promotionTy = getPromotionType(e->getSubExpr()->getType());
  mlir::Value result = VisitRealImag(e, promotionTy);
  if (result && !promotionTy.isNull())
    result = emitUnPromotedValue(result, e->getType());
  return result;
}

mlir::Value ScalarExprEmitter::VisitRealImag(const UnaryOperator *e,
                                             QualType promotionTy) {
  assert(e->getOpcode() == clang::UO_Real ||
         e->getOpcode() == clang::UO_Imag &&
             "Invalid UnaryOp kind for ComplexType Real or Imag");

  Expr *op = e->getSubExpr();
  mlir::Location loc = cgf.getLoc(e->getExprLoc());
  if (op->getType()->isAnyComplexType()) {
    // If it's an l-value, load through the appropriate subobject l-value.
    // Note that we have to ask `e` because `op` might be an l-value that
    // this won't work for, e.g. an Obj-C property
    mlir::Value complex = cgf.emitComplexExpr(op);
    if (e->isGLValue() && !promotionTy.isNull()) {
      promotionTy = promotionTy->isAnyComplexType()
                        ? promotionTy
                        : cgf.getContext().getComplexType(promotionTy);
      complex = cgf.emitPromotedValue(complex, promotionTy);
    }

    return e->getOpcode() == clang::UO_Real
               ? builder.createComplexReal(loc, complex)
               : builder.createComplexImag(loc, complex);
  }

  if (e->getOpcode() == UO_Real) {
    mlir::Value operand = promotionTy.isNull()
                              ? Visit(op)
                              : cgf.emitPromotedScalarExpr(op, promotionTy);
    return builder.createComplexReal(loc, operand);
  }

  // __imag on a scalar returns zero. Emit the subexpr to ensure side
  // effects are evaluated, but not the actual value.
  mlir::Value operand;
  if (op->isGLValue()) {
    operand = cgf.emitLValue(op).getPointer();
    operand = cir::LoadOp::create(builder, loc, operand);
  } else if (!promotionTy.isNull()) {
    operand = cgf.emitPromotedScalarExpr(op, promotionTy);
  } else {
    operand = cgf.emitScalarExpr(op);
  }
  return builder.createComplexImag(loc, operand);
}

/// Return the size or alignment of the type of argument of the sizeof
/// expression as an integer.
mlir::Value ScalarExprEmitter::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *e) {
  const QualType typeToSize = e->getTypeOfArgument();
  const mlir::Location loc = cgf.getLoc(e->getSourceRange());
  if (auto kind = e->getKind();
      kind == UETT_SizeOf || kind == UETT_DataSizeOf) {
    if (cgf.getContext().getAsVariableArrayType(typeToSize)) {
      cgf.getCIRGenModule().errorNYI(e->getSourceRange(),
                                     "sizeof operator for VariableArrayType",
                                     e->getStmtClassName());
      return builder.getConstant(
          loc, cir::IntAttr::get(cgf.cgm.UInt64Ty,
                                 llvm::APSInt(llvm::APInt(64, 1), true)));
    }
  } else if (e->getKind() == UETT_OpenMPRequiredSimdAlign) {
    cgf.getCIRGenModule().errorNYI(
        e->getSourceRange(), "sizeof operator for OpenMpRequiredSimdAlign",
        e->getStmtClassName());
    return builder.getConstant(
        loc, cir::IntAttr::get(cgf.cgm.UInt64Ty,
                               llvm::APSInt(llvm::APInt(64, 1), true)));
  }

  return builder.getConstant(
      loc, cir::IntAttr::get(cgf.cgm.UInt64Ty,
                             e->EvaluateKnownConstInt(cgf.getContext())));
}

/// Return true if the specified expression is cheap enough and side-effect-free
/// enough to evaluate unconditionally instead of conditionally.  This is used
/// to convert control flow into selects in some cases.
/// TODO(cir): can be shared with LLVM codegen.
static bool isCheapEnoughToEvaluateUnconditionally(const Expr *e,
                                                   CIRGenFunction &cgf) {
  // Anything that is an integer or floating point constant is fine.
  return e->IgnoreParens()->isEvaluatable(cgf.getContext());

  // Even non-volatile automatic variables can't be evaluated unconditionally.
  // Referencing a thread_local may cause non-trivial initialization work to
  // occur. If we're inside a lambda and one of the variables is from the scope
  // outside the lambda, that function may have returned already. Reading its
  // locals is a bad idea. Also, these reads may introduce races there didn't
  // exist in the source-level program.
}

mlir::Value ScalarExprEmitter::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *e) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(e->getSourceRange());
  ignoreResultAssign = false;

  // Bind the common expression if necessary.
  CIRGenFunction::OpaqueValueMapping binding(cgf, e);

  Expr *condExpr = e->getCond();
  Expr *lhsExpr = e->getTrueExpr();
  Expr *rhsExpr = e->getFalseExpr();

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm.
  bool condExprBool;
  if (cgf.constantFoldsToBool(condExpr, condExprBool)) {
    Expr *live = lhsExpr, *dead = rhsExpr;
    if (!condExprBool)
      std::swap(live, dead);

    // If the dead side doesn't have labels we need, just emit the Live part.
    if (!cgf.containsLabel(dead)) {
      if (condExprBool)
        assert(!cir::MissingFeatures::incrementProfileCounter());
      mlir::Value result = Visit(live);

      // If the live part is a throw expression, it acts like it has a void
      // type, so evaluating it returns a null Value.  However, a conditional
      // with non-void type must return a non-null Value.
      if (!result && !e->getType()->isVoidType()) {
        cgf.cgm.errorNYI(e->getSourceRange(),
                         "throw expression in conditional operator");
        result = {};
      }

      return result;
    }
  }

  QualType condType = condExpr->getType();

  // OpenCL: If the condition is a vector, we can treat this condition like
  // the select function.
  if ((cgf.getLangOpts().OpenCL && condType->isVectorType()) ||
      condType->isExtVectorType()) {
    assert(!cir::MissingFeatures::vectorType());
    cgf.cgm.errorNYI(e->getSourceRange(), "vector ternary op");
  }

  if (condType->isVectorType() || condType->isSveVLSBuiltinType()) {
    if (!condType->isVectorType()) {
      assert(!cir::MissingFeatures::vecTernaryOp());
      cgf.cgm.errorNYI(loc, "TernaryOp for SVE vector");
      return {};
    }

    mlir::Value condValue = Visit(condExpr);
    mlir::Value lhsValue = Visit(lhsExpr);
    mlir::Value rhsValue = Visit(rhsExpr);
    return builder.create<cir::VecTernaryOp>(loc, condValue, lhsValue,
                                             rhsValue);
  }

  // If this is a really simple expression (like x ? 4 : 5), emit this as a
  // select instead of as control flow.  We can only do this if it is cheap
  // and safe to evaluate the LHS and RHS unconditionally.
  if (isCheapEnoughToEvaluateUnconditionally(lhsExpr, cgf) &&
      isCheapEnoughToEvaluateUnconditionally(rhsExpr, cgf)) {
    bool lhsIsVoid = false;
    mlir::Value condV = cgf.evaluateExprAsBool(condExpr);
    assert(!cir::MissingFeatures::incrementProfileCounter());

    mlir::Value lhs = Visit(lhsExpr);
    if (!lhs) {
      lhs = builder.getNullValue(cgf.VoidTy, loc);
      lhsIsVoid = true;
    }

    mlir::Value rhs = Visit(rhsExpr);
    if (lhsIsVoid) {
      assert(!rhs && "lhs and rhs types must match");
      rhs = builder.getNullValue(cgf.VoidTy, loc);
    }

    return builder.createSelect(loc, condV, lhs, rhs);
  }

  mlir::Value condV = cgf.emitOpOnBoolExpr(loc, condExpr);
  CIRGenFunction::ConditionalEvaluation eval(cgf);
  SmallVector<mlir::OpBuilder::InsertPoint, 2> insertPoints{};
  mlir::Type yieldTy{};

  auto emitBranch = [&](mlir::OpBuilder &b, mlir::Location loc, Expr *expr) {
    CIRGenFunction::LexicalScope lexScope{cgf, loc, b.getInsertionBlock()};
    cgf.curLexScope->setAsTernary();

    assert(!cir::MissingFeatures::incrementProfileCounter());
    eval.beginEvaluation();
    mlir::Value branch = Visit(expr);
    eval.endEvaluation();

    if (branch) {
      yieldTy = branch.getType();
      b.create<cir::YieldOp>(loc, branch);
    } else {
      // If LHS or RHS is a throw or void expression we need to patch
      // arms as to properly match yield types.
      insertPoints.push_back(b.saveInsertionPoint());
    }
  };

  mlir::Value result = builder
                           .create<cir::TernaryOp>(
                               loc, condV,
                               /*trueBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 emitBranch(b, loc, lhsExpr);
                               },
                               /*falseBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 emitBranch(b, loc, rhsExpr);
                               })
                           .getResult();

  if (!insertPoints.empty()) {
    // If both arms are void, so be it.
    if (!yieldTy)
      yieldTy = cgf.VoidTy;

    // Insert required yields.
    for (mlir::OpBuilder::InsertPoint &toInsert : insertPoints) {
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
  }

  return result;
}

mlir::Value CIRGenFunction::emitScalarPrePostIncDec(const UnaryOperator *e,
                                                    LValue lv,
                                                    cir::UnaryOpKind kind,
                                                    bool isPre) {
  return ScalarExprEmitter(*this, builder)
      .emitScalarPrePostIncDec(e, lv, kind, isPre);
}
