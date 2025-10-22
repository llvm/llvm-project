//===--- CIRGenException.cpp - Emit CIR Code for C++ exceptions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ exception related code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace clang::CIRGen;

void CIRGenFunction::emitCXXThrowExpr(const CXXThrowExpr *e) {
  const llvm::Triple &triple = getTarget().getTriple();
  if (cgm.getLangOpts().OpenMPIsTargetDevice &&
      (triple.isNVPTX() || triple.isAMDGCN())) {
    cgm.errorNYI("emitCXXThrowExpr OpenMP with NVPTX or AMDGCN Triples");
    return;
  }

  if (const Expr *subExpr = e->getSubExpr()) {
    QualType throwType = subExpr->getType();
    if (throwType->isObjCObjectPointerType()) {
      cgm.errorNYI("emitCXXThrowExpr ObjCObjectPointerType");
      return;
    }

    cgm.getCXXABI().emitThrow(*this, e);
    return;
  }

  cgm.getCXXABI().emitRethrow(*this, /*isNoReturn=*/true);
}

void CIRGenFunction::emitAnyExprToExn(const Expr *e, Address addr) {
  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  assert(!cir::MissingFeatures::ehCleanupScope());

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  mlir::Type ty = convertTypeForMem(e->getType());
  Address typedAddr = addr.withElementType(builder, ty);

  // From LLVM's codegen:
  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  emitAnyExprToMem(e, typedAddr, e->getType().getQualifiers(),
                   /*isInitializer=*/true);

  // Deactivate the cleanup block.
  assert(!cir::MissingFeatures::ehCleanupScope());
}

mlir::LogicalResult CIRGenFunction::emitCXXTryStmt(const CXXTryStmt &s) {
  if (s.getTryBlock()->body_empty())
    return mlir::LogicalResult::success();

  mlir::Location loc = getLoc(s.getSourceRange());
  // Create a scope to hold try local storage for catch params.

  mlir::OpBuilder::InsertPoint scopeIP;
  cir::ScopeOp::create(
      builder, loc,
      /*scopeBuilder=*/[&](mlir::OpBuilder &b, mlir::Location loc) {
        scopeIP = builder.saveInsertionPoint();
      });

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.restoreInsertionPoint(scopeIP);
  mlir::LogicalResult result = emitCXXTryStmtUnderScope(s);
  cir::YieldOp::create(builder, loc);
  return result;
}

mlir::LogicalResult
CIRGenFunction::emitCXXTryStmtUnderScope(const CXXTryStmt &s) {
  const llvm::Triple &t = getTarget().getTriple();
  // If we encounter a try statement on in an OpenMP target region offloaded to
  // a GPU, we treat it as a basic block.
  const bool isTargetDevice =
      (cgm.getLangOpts().OpenMPIsTargetDevice && (t.isNVPTX() || t.isAMDGCN()));
  if (isTargetDevice) {
    cgm.errorNYI(
        "emitCXXTryStmtUnderScope: OpenMP target region offloaded to GPU");
    return mlir::success();
  }

  unsigned numHandlers = s.getNumHandlers();
  mlir::Location tryLoc = getLoc(s.getBeginLoc());
  mlir::OpBuilder::InsertPoint beginInsertTryBody;

  bool hasCatchAll = false;
  for (unsigned i = 0; i != numHandlers; ++i) {
    hasCatchAll |= s.getHandler(i)->getExceptionDecl() == nullptr;
    if (hasCatchAll)
      break;
  }

  // Create the scope to represent only the C/C++ `try {}` part. However,
  // don't populate right away. Create regions for the catch handlers,
  // but don't emit the handler bodies yet. For now, only make sure the
  // scope returns the exception information.
  auto tryOp = cir::TryOp::create(
      builder, tryLoc,
      /*tryBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        beginInsertTryBody = builder.saveInsertionPoint();
      },
      /*handlersBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::OperationState &result) {
        mlir::OpBuilder::InsertionGuard guard(b);

        // We create an extra region for an unwind catch handler in case the
        // catch-all handler doesn't exists
        unsigned numRegionsToCreate =
            hasCatchAll ? numHandlers : numHandlers + 1;

        for (unsigned i = 0; i != numRegionsToCreate; ++i) {
          mlir::Region *region = result.addRegion();
          builder.createBlock(region);
        }
      });

  // Finally emit the body for try/catch.
  {
    mlir::Location loc = tryOp.getLoc();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(beginInsertTryBody);
    CIRGenFunction::LexicalScope tryScope{*this, loc,
                                          builder.getInsertionBlock()};

    tryScope.setAsTry(tryOp);

    // Attach the basic blocks for the catch regions.
    enterCXXTryStmt(s, tryOp);

    // Emit the body for the `try {}` part.
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      CIRGenFunction::LexicalScope tryBodyScope{*this, loc,
                                                builder.getInsertionBlock()};
      if (emitStmt(s.getTryBlock(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    }

    // Emit catch clauses.
    exitCXXTryStmt(s);
  }

  return mlir::success();
}

void CIRGenFunction::enterCXXTryStmt(const CXXTryStmt &s, cir::TryOp tryOp,
                                     bool isFnTryBlock) {
  unsigned numHandlers = s.getNumHandlers();
  EHCatchScope *catchScope = ehStack.pushCatch(numHandlers);
  for (unsigned i = 0; i != numHandlers; ++i) {
    const CXXCatchStmt *catchStmt = s.getHandler(i);
    if (catchStmt->getExceptionDecl()) {
      cgm.errorNYI("enterCXXTryStmt: CatchStmt with ExceptionDecl");
      return;
    }

    // No exception decl indicates '...', a catch-all.
    mlir::Region *handler = &tryOp.getHandlerRegions()[i];
    catchScope->setHandler(i, cgm.getCXXABI().getCatchAllTypeInfo(), handler);

    // Under async exceptions, catch(...) needs to catch HW exception too
    // Mark scope with SehTryBegin as a SEH __try scope
    if (getLangOpts().EHAsynch) {
      cgm.errorNYI("enterCXXTryStmt: EHAsynch");
      return;
    }
  }
}

void CIRGenFunction::exitCXXTryStmt(const CXXTryStmt &s, bool isFnTryBlock) {
  unsigned numHandlers = s.getNumHandlers();
  EHCatchScope &catchScope = cast<EHCatchScope>(*ehStack.begin());
  assert(catchScope.getNumHandlers() == numHandlers);
  cir::TryOp tryOp = curLexScope->getTry();

  // If the catch was not required, bail out now.
  if (!catchScope.mayThrow()) {
    catchScope.clearHandlerBlocks();
    ehStack.popCatch();

    // Drop all basic block from all catch regions.
    SmallVector<mlir::Block *> eraseBlocks;
    for (mlir::Region &handlerRegion : tryOp.getHandlerRegions()) {
      if (handlerRegion.empty())
        continue;

      for (mlir::Block &b : handlerRegion.getBlocks())
        eraseBlocks.push_back(&b);
    }

    for (mlir::Block *b : eraseBlocks)
      b->erase();

    tryOp.setHandlerTypesAttr({});
    return;
  }

  cgm.errorNYI("exitCXXTryStmt: Required catch");
}
