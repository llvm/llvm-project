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
  ScalarExprEmitter(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                    bool ira = false)
      : cgf(cgf), builder(builder), ignoreResultAssign(ira) {}

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

  mlir::Value VisitCastExpr(CastExpr *E);

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
  mlir::Value emitScalarConversion(mlir::Value src, QualType srcType,
                                   QualType dstType, SourceLocation loc) {
    // No sort of type conversion is implemented yet, but the path for implicit
    // paths goes through here even if the type isn't being changed.
    srcType = srcType.getCanonicalType();
    dstType = dstType.getCanonicalType();
    if (srcType == dstType)
      return src;

    cgf.getCIRGenModule().errorNYI(loc,
                                   "emitScalarConversion for unequal types");
    return {};
  }
};

} // namespace

/// Emit the computation of the specified expression of scalar type.
mlir::Value CIRGenFunction::emitScalarExpr(const Expr *e) {
  assert(e && hasScalarEvaluationKind(e->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(e));
}

// Emit code for an explicit or implicit cast.  Implicit
// casts have to handle a more broad range of conversions than explicit
// casts, as they handle things like function to ptr-to-function decay
// etc.
mlir::Value ScalarExprEmitter::VisitCastExpr(CastExpr *ce) {
  Expr *e = ce->getSubExpr();
  QualType destTy = ce->getType();
  CastKind kind = ce->getCastKind();

  switch (kind) {
  case CK_LValueToRValue:
    assert(cgf.getContext().hasSameUnqualifiedType(e->getType(), destTy));
    assert(e->isGLValue() && "lvalue-to-rvalue applied to r-value!");
    return Visit(const_cast<Expr *>(e));

  case CK_IntegralCast: {
    assert(!cir::MissingFeatures::scalarConversionOpts());
    return emitScalarConversion(Visit(e), e->getType(), destTy,
                                ce->getExprLoc());
  }

  default:
    cgf.getCIRGenModule().errorNYI(e->getSourceRange(),
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
