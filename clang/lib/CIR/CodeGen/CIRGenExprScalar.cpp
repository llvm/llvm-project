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

#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"

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

  mlir::Value VisitIntegerLiteral(const IntegerLiteral *e) {
    mlir::Type type = cgf.convertType(e->getType());
    return builder.create<cir::ConstantOp>(
        cgf.getLoc(e->getExprLoc()), type,
        builder.getAttr<cir::IntAttr>(type, e->getValue()));
  }

  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *e) {
    mlir::Type type = cgf.convertType(e->getType());
    return builder.create<cir::ConstantOp>(
        cgf.getLoc(e->getExprLoc()), type,
        builder.getCIRBoolAttr(e->getValue()));
  }
};
} // namespace

/// Emit the computation of the specified expression of scalar type.
mlir::Value CIRGenFunction::emitScalarExpr(const Expr *e) {
  assert(e && hasScalarEvaluationKind(e->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(e));
}
