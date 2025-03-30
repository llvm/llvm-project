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
    auto lhsci = dyn_cast<cir::ConstantOp>(lhs.getDefiningOp());
    auto rhsci = dyn_cast<cir::ConstantOp>(rhs.getDefiningOp());
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

  mlir::Value emitPromotedValue(mlir::Value result, QualType promotionType) {
    cgf.cgm.errorNYI(result.getLoc(), "floating cast for promoted value");
    return {};
  }

  mlir::Value emitUnPromotedValue(mlir::Value result, QualType exprType) {
    cgf.cgm.errorNYI(result.getLoc(), "floating cast for unpromoted value");
    return {};
  }

  mlir::Value emitPromoted(const Expr *e, QualType promotionType);

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

  mlir::Value emitLoadOfLValue(LValue lv, SourceLocation loc) {
    return cgf.emitLoadOfLValue(lv, loc).getScalarVal();
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
    } else if (isa<PointerType>(type)) {
      cgf.cgm.errorNYI(e->getSourceRange(), "Unary inc/dec pointer");
      return {};
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
    if (ty->getAs<ComplexType>()) {
      assert(!cir::MissingFeatures::complexType());
      cgf.cgm.errorNYI("promotion to complex type");
      return QualType();
    }
    if (ty.UseExcessPrecision(cgf.getContext())) {
      if (ty->getAs<VectorType>()) {
        assert(!cir::MissingFeatures::vectorType());
        cgf.cgm.errorNYI("promotion to vector type");
        return QualType();
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
};

LValue ScalarExprEmitter::emitCompoundAssignLValue(
    const CompoundAssignOperator *e,
    mlir::Value (ScalarExprEmitter::*func)(const BinOpInfo &),
    mlir::Value &result) {
  QualType lhsTy = e->getLHS()->getType();
  BinOpInfo opInfo;

  if (e->getComputationResultType()->isAnyComplexType()) {
    cgf.cgm.errorNYI(result.getLoc(), "complex lvalue assign");
    return LValue();
  }

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
    cgf.cgm.errorNYI(e->getSourceRange(), "store through bitfield lvalue");
  else
    cgf.emitStoreThroughLValue(RValue::get(result), lhsLV);

  if (cgf.getLangOpts().OpenMP)
    cgf.cgm.errorNYI(e->getSourceRange(), "openmp");

  return lhsLV;
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
  } else if (isa<UnaryOperator>(e)) {
    cgf.cgm.errorNYI(e->getSourceRange(), "unary operators");
    return {};
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

} // namespace

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
  // If a null pointer expression's type is the C++0x nullptr_t, then
  // it's not necessarily a simple constant and it must be evaluated
  // for its potential side effects.
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
  cgf.cgm.errorNYI(op.loc, "pointer arithmetic");
  return {};
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

  if (cir::isFPOrFPVectorTy(ops.lhs.getType())) {
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

  if (cir::isFPOrFPVectorTy(ops.lhs.getType())) {
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

    if (cir::isFPOrFPVectorTy(ops.lhs.getType())) {
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

  cgf.cgm.errorNYI("shift ops");
  return {};
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
  cgf.cgm.errorNYI("shift ops");
  return {};
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

mlir::Value CIRGenFunction::emitScalarConversion(mlir::Value src,
                                                 QualType srcTy, QualType dstTy,
                                                 SourceLocation loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(srcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(dstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this, builder)
      .emitScalarConversion(src, srcTy, dstTy, loc);
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

mlir::Value CIRGenFunction::emitScalarPrePostIncDec(const UnaryOperator *e,
                                                    LValue lv, bool isInc,
                                                    bool isPre) {
  return ScalarExprEmitter(*this, builder)
      .emitScalarPrePostIncDec(e, lv, isInc, isPre);
}
