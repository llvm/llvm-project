//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"

#include "mlir/IR/Builders.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

void CIRGenFunction::emitCompoundStmtWithoutScope(const CompoundStmt &s) {
  for (auto *curStmt : s.body()) {
    if (emitStmt(curStmt, /*useCurrentScope=*/false).failed())
      getCIRGenModule().errorNYI(curStmt->getSourceRange(), "statement");
  }
}

void CIRGenFunction::emitCompoundStmt(const CompoundStmt &s) {
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
  auto scope = builder.create<cir::ScopeOp>(
      scopeLoc, [&](mlir::OpBuilder &b, mlir::Type &type, mlir::Location loc) {
        emitCompoundStmtWithoutScope(s);
      });

  // This code to insert a cir.yield at the end of the scope is temporary until
  // CIRGenFunction::LexicalScope::cleanup() is upstreamed.
  if (!scope.getRegion().empty()) {
    mlir::Block &lastBlock = scope.getRegion().back();
    if (lastBlock.empty() || !lastBlock.mightHaveTerminator() ||
        !lastBlock.getTerminator()->hasTrait<mlir::OpTrait::IsTerminator>()) {
      builder.setInsertionPointToEnd(&lastBlock);
      builder.create<cir::YieldOp>(getLoc(s.getEndLoc()));
    }
  }
}

// Build CIR for a statement. useCurrentScope should be true if no new scopes
// need to be created when finding a compound statement.
mlir::LogicalResult CIRGenFunction::emitStmt(const Stmt *s,
                                             bool useCurrentScope,
                                             ArrayRef<const Attr *> attr) {
  if (mlir::succeeded(emitSimpleStmt(s, useCurrentScope)))
    return mlir::success();

  // Only a subset of simple statements are supported at the moment.  When more
  // kinds of statements are supported, a
  //     switch (s->getStmtClass()) {
  // will be added here.
  return mlir::failure();
}

mlir::LogicalResult CIRGenFunction::emitSimpleStmt(const Stmt *s,
                                                   bool useCurrentScope) {
  switch (s->getStmtClass()) {
  default:
    // Only compound and return statements are supported right now.
    return mlir::failure();
  case Stmt::DeclStmtClass:
    return emitDeclStmt(cast<DeclStmt>(*s));
  case Stmt::CompoundStmtClass:
    if (useCurrentScope)
      emitCompoundStmtWithoutScope(cast<CompoundStmt>(*s));
    else
      emitCompoundStmt(cast<CompoundStmt>(*s));
    break;
  case Stmt::ReturnStmtClass:
    return emitReturnStmt(cast<ReturnStmt>(*s));
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitDeclStmt(const DeclStmt &s) {
  assert(builder.getInsertionBlock() && "expected valid insertion point");

  for (const Decl *I : s.decls())
    emitDecl(*I);

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitReturnStmt(const ReturnStmt &s) {
  mlir::Location loc = getLoc(s.getSourceRange());
  const Expr *rv = s.getRetValue();

  if (getContext().getLangOpts().ElideConstructors && s.getNRVOCandidate() &&
      s.getNRVOCandidate()->isNRVOVariable()) {
    getCIRGenModule().errorNYI(s.getSourceRange(),
                               "named return value optimization");
  } else if (!rv) {
    // No return expression. Do nothing.
    // TODO(CIR): In the future when function returns are fully implemented,
    // this section will do nothing.  But for now a ReturnOp is necessary.
    builder.create<ReturnOp>(loc);
  } else if (rv->getType()->isVoidType()) {
    // No return value. Emit the return expression for its side effects.
    // TODO(CIR): Once emitAnyExpr(e) has been upstreamed, get rid of the check
    // and just call emitAnyExpr(rv) here.
    if (CIRGenFunction::hasScalarEvaluationKind(rv->getType())) {
      emitScalarExpr(rv);
    } else {
      getCIRGenModule().errorNYI(s.getSourceRange(),
                                 "non-scalar function return type");
    }
    builder.create<ReturnOp>(loc);
  } else if (fnRetTy->isReferenceType()) {
    getCIRGenModule().errorNYI(s.getSourceRange(),
                               "function return type that is a reference");
  } else {
    mlir::Value value = nullptr;
    switch (CIRGenFunction::getEvaluationKind(rv->getType())) {
    case cir::TEK_Scalar:
      value = emitScalarExpr(rv);
      if (value) { // Change this to an assert once emitScalarExpr is complete
        builder.create<ReturnOp>(loc, llvm::ArrayRef(value));
      }
      break;
    default:
      getCIRGenModule().errorNYI(s.getSourceRange(),
                                 "non-scalar function return type");
      break;
    }
  }

  return mlir::success();
}
