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

#include "mlir/IR/Value.h"

#include <cassert>

using namespace clang;
using namespace clang::CIRGen;

namespace {

class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
  CIRGenFunction &cgf;
  CIRGenBuilderTy &builder;
  bool ignoreResultAssign;

public:
  ScalarExprEmitter(CIRGenFunction &cgf, CIRGenBuilderTy &builder)
      : cgf(cgf), builder(builder) {}

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

  /// Emits the address of the l-value, then loads and returns the result.
  mlir::Value emitLoadOfLValue(const Expr *e) {
    LValue lv = cgf.emitLValue(e);
    // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
    return cgf.emitLoadOfLValue(lv, e->getExprLoc()).getScalarVal();
  }

  // l-values
  mlir::Value VisitDeclRefExpr(DeclRefExpr *e) {
    assert(!cir::MissingFeatures::tryEmitAsConstant());
    return emitLoadOfLValue(e);
  }

  mlir::Value VisitIntegerLiteral(const IntegerLiteral *e) {
    mlir::Type type = cgf.convertType(e->getType());
    return builder.create<cir::ConstantOp>(
        cgf.getLoc(e->getExprLoc()), type,
        builder.getAttr<cir::IntAttr>(type, e->getValue()));
  }

  mlir::Value VisitFloatingLiteral(const FloatingLiteral *e) {
    mlir::Type type = cgf.convertType(e->getType());
    assert(mlir::isa<cir::CIRFPTypeInterface>(type) &&
           "expect floating-point type");
    return builder.create<cir::ConstantOp>(
        cgf.getLoc(e->getExprLoc()), type,
        builder.getAttr<cir::FPAttr>(type, e->getValue()));
  }

  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *e) {
    mlir::Type type = cgf.convertType(e->getType());
    return builder.create<cir::ConstantOp>(
        cgf.getLoc(e->getExprLoc()), type,
        builder.getCIRBoolAttr(e->getValue()));
  }

  mlir::Value VisitCastExpr(CastExpr *e);

  mlir::Value VisitExplicitCastExpr(ExplicitCastExpr *e) {
    return VisitCastExpr(e);
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
      mlir::Type boolType = builder.getBoolTy();
      return builder.create<cir::ConstantOp>(loc, boolType,
                                             builder.getCIRBoolAttr(false));
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
    assert(!cir::MissingFeatures::vectorType());

    std::optional<cir::CastKind> castKind;

    if (mlir::isa<cir::BoolType>(srcTy)) {
      if (opts.treatBooleanAsSigned)
        cgf.getCIRGenModule().errorNYI("signed bool");
      if (cgf.getBuilder().isInt(dstTy))
        castKind = cir::CastKind::bool_to_int;
      else if (mlir::isa<cir::CIRFPTypeInterface>(dstTy))
        castKind = cir::CastKind::bool_to_float;
      else
        llvm_unreachable("Internal error: Cast to unexpected type");
    } else if (cgf.getBuilder().isInt(srcTy)) {
      if (cgf.getBuilder().isInt(dstTy))
        castKind = cir::CastKind::integral;
      else if (mlir::isa<cir::CIRFPTypeInterface>(dstTy))
        castKind = cir::CastKind::int_to_float;
      else
        llvm_unreachable("Internal error: Cast to unexpected type");
    } else if (mlir::isa<cir::CIRFPTypeInterface>(srcTy)) {
      if (cgf.getBuilder().isInt(dstTy)) {
        // If we can't recognize overflow as undefined behavior, assume that
        // overflow saturates. This protects against normal optimizations if we
        // are compiling with non-standard FP semantics.
        if (!cgf.cgm.getCodeGenOpts().StrictFloatCastOverflow)
          cgf.getCIRGenModule().errorNYI("strict float cast overflow");
        assert(!cir::MissingFeatures::fpConstraints());
        castKind = cir::CastKind::float_to_int;
      } else if (mlir::isa<cir::CIRFPTypeInterface>(dstTy)) {
        cgf.getCIRGenModule().errorNYI("floating point casts");
        return cgf.createDummyValue(src.getLoc(), dstType);
      } else {
        llvm_unreachable("Internal error: Cast to unexpected type");
      }
    } else {
      llvm_unreachable("Internal error: Cast from unexpected type");
    }

    assert(castKind.has_value() && "Internal error: CastKind not set.");
    return builder.create<cir::CastOp>(src.getLoc(), fullDstTy, *castKind, src);
  }

  mlir::Value VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *e);

  // Unary Operators.
  mlir::Value VisitUnaryPostDec(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, false, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, true, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, false, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return emitScalarPrePostIncDec(e, lv, true, true);
  }
  mlir::Value emitScalarPrePostIncDec(const UnaryOperator *e, LValue lv,
                                      bool isInc, bool isPre) {
    if (cgf.getLangOpts().OpenMP)
      cgf.cgm.errorNYI(e->getSourceRange(), "inc/dec OpenMP");

    QualType type = e->getSubExpr()->getType();

    mlir::Value value;
    mlir::Value input;

    if (type->getAs<AtomicType>()) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Atomic inc/dec");
      // TODO(cir): This is not correct, but it will produce reasonable code
      // until atomic operations are implemented.
      value = cgf.emitLoadOfLValue(lv, e->getExprLoc()).getScalarVal();
      input = value;
    } else {
      value = cgf.emitLoadOfLValue(lv, e->getExprLoc()).getScalarVal();
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
      value = builder.create<cir::ConstantOp>(cgf.getLoc(e->getExprLoc()),
                                              cgf.convertType(type),
                                              builder.getCIRBoolAttr(true));
    } else if (type->isIntegerType()) {
      QualType promotedType;
      bool canPerformLossyDemotionCheck = false;
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
        value = emitIncDecConsiderOverflowBehavior(e, value, isInc);
      } else {
        cir::UnaryOpKind kind =
            e->isIncrementOp() ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec;
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        value = emitUnaryOp(e, kind, input);
      }
    } else if (const PointerType *ptr = type->getAs<PointerType>()) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec pointer");
      return {};
    } else if (type->isVectorType()) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec vector");
      return {};
    } else if (type->isRealFloatingType()) {
      assert(!cir::MissingFeatures::CGFPOptionsRAII());

      if (type->isHalfType() &&
          !cgf.getContext().getLangOpts().NativeHalfType) {
        cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec half");
        return {};
      }

      if (mlir::isa<cir::SingleType, cir::DoubleType>(value.getType())) {
        // Create the inc/dec operation.
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        cir::UnaryOpKind kind =
            (isInc ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec);
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
    if (lv.isBitField()) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec bitfield");
      return {};
    } else {
      cgf.emitStoreThroughLValue(RValue::get(value), lv);
    }

    // If this is a postinc, return the value read from memory, otherwise use
    // the updated value.
    return isPre ? value : input;
  }

  mlir::Value emitIncDecConsiderOverflowBehavior(const UnaryOperator *e,
                                                 mlir::Value inVal,
                                                 bool isInc) {
    assert(!cir::MissingFeatures::opUnarySignedOverflow());
    cir::UnaryOpKind kind =
        e->isIncrementOp() ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec;
    switch (cgf.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      return emitUnaryOp(e, kind, inVal);
    case LangOptions::SOB_Undefined:
      assert(!cir::MissingFeatures::sanitizers());
      return emitUnaryOp(e, kind, inVal);
      break;
    case LangOptions::SOB_Trapping:
      if (!e->canOverflow())
        return emitUnaryOp(e, kind, inVal);
      cgf.cgm.errorNYI(e->getSourceRange(), "inc/def overflow SOB_Trapping");
      return {};
    }
    llvm_unreachable("Unexpected signed overflow behavior kind");
  }

  mlir::Value VisitUnaryPlus(const UnaryOperator *e,
                             QualType promotionType = QualType()) {
    if (!promotionType.isNull())
      cgf.cgm.errorNYI(e->getSourceRange(), "VisitUnaryPlus: promotionType");
    assert(!cir::MissingFeatures::opUnaryPromotionType());
    mlir::Value result = emitUnaryPlusOrMinus(e, cir::UnaryOpKind::Plus);
    return result;
  }

  mlir::Value VisitUnaryMinus(const UnaryOperator *e,
                              QualType promotionType = QualType()) {
    if (!promotionType.isNull())
      cgf.cgm.errorNYI(e->getSourceRange(), "VisitUnaryMinus: promotionType");
    assert(!cir::MissingFeatures::opUnaryPromotionType());
    mlir::Value result = emitUnaryPlusOrMinus(e, cir::UnaryOpKind::Minus);
    return result;
  }

  mlir::Value emitUnaryPlusOrMinus(const UnaryOperator *e,
                                   cir::UnaryOpKind kind) {
    ignoreResultAssign = false;

    assert(!cir::MissingFeatures::opUnaryPromotionType());
    mlir::Value operand = Visit(e->getSubExpr());

    assert(!cir::MissingFeatures::opUnarySignedOverflow());

    // NOTE: LLVM codegen will lower this directly to either a FNeg
    // or a Sub instruction.  In CIR this will be handled later in LowerToLLVM.
    return emitUnaryOp(e, kind, operand);
  }

  mlir::Value emitUnaryOp(const UnaryOperator *e, cir::UnaryOpKind kind,
                          mlir::Value input) {
    return builder.create<cir::UnaryOp>(
        cgf.getLoc(e->getSourceRange().getBegin()), input.getType(), kind,
        input);
  }

  mlir::Value VisitUnaryNot(const UnaryOperator *e) {
    ignoreResultAssign = false;
    mlir::Value op = Visit(e->getSubExpr());
    return emitUnaryOp(e, cir::UnaryOpKind::Not, op);
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
      if (mlir::isa<cir::CIRFPTypeInterface>(mlirDstType)) {
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
};

} // namespace

/// Emit the computation of the specified expression of scalar type.
mlir::Value CIRGenFunction::emitScalarExpr(const Expr *e) {
  assert(e && hasScalarEvaluationKind(e->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(e));
}

[[maybe_unused]] static bool MustVisitNullValue(const Expr *e) {
  // If a null pointer expression's type is the C++0x nullptr_t, then
  // it's not necessarily a simple constant and it must be evaluated
  // for its potential side effects.
  return e->getType()->isNullPtrType();
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

  case CK_NullToPointer: {
    if (MustVisitNullValue(subExpr))
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
    cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(), "fp options");
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

  default:
    cgf.getCIRGenModule().errorNYI(subExpr->getSourceRange(),
                                   "CastExpr: ", ce->getCastKindName());
  }
  return {};
}

/// Return the size or alignment of the type of argument of the sizeof
/// expression as an integer.
mlir::Value ScalarExprEmitter::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *e) {
  const QualType typeToSize = e->getTypeOfArgument();
  const mlir::Location loc = cgf.getLoc(e->getSourceRange());
  if (auto kind = e->getKind();
      kind == UETT_SizeOf || kind == UETT_DataSizeOf) {
    if (const VariableArrayType *variableArrTy =
            cgf.getContext().getAsVariableArrayType(typeToSize)) {
      cgf.getCIRGenModule().errorNYI(e->getSourceRange(),
                                     "sizeof operator for VariableArrayType",
                                     e->getStmtClassName());
      return builder.getConstant(
          loc, builder.getAttr<cir::IntAttr>(
                   cgf.cgm.UInt64Ty, llvm::APSInt(llvm::APInt(64, 1), true)));
    }
  } else if (e->getKind() == UETT_OpenMPRequiredSimdAlign) {
    cgf.getCIRGenModule().errorNYI(
        e->getSourceRange(), "sizeof operator for OpenMpRequiredSimdAlign",
        e->getStmtClassName());
    return builder.getConstant(
        loc, builder.getAttr<cir::IntAttr>(
                 cgf.cgm.UInt64Ty, llvm::APSInt(llvm::APInt(64, 1), true)));
  } else if (e->getKind() == UETT_VectorElements) {
    cgf.getCIRGenModule().errorNYI(e->getSourceRange(),
                                   "sizeof operator for VectorElements",
                                   e->getStmtClassName());
    return builder.getConstant(
        loc, builder.getAttr<cir::IntAttr>(
                 cgf.cgm.UInt64Ty, llvm::APSInt(llvm::APInt(64, 1), true)));
  }

  return builder.getConstant(
      loc, builder.getAttr<cir::IntAttr>(
               cgf.cgm.UInt64Ty, e->EvaluateKnownConstInt(cgf.getContext())));
}

mlir::Value CIRGenFunction::emitScalarPrePostIncDec(const UnaryOperator *E,
                                                    LValue LV, bool isInc,
                                                    bool isPre) {
  return ScalarExprEmitter(*this, builder)
      .emitScalarPrePostIncDec(E, LV, isInc, isPre);
}
