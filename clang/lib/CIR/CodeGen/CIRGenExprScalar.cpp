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
