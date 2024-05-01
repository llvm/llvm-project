//===--- CIRGenStmtOpenMP.cpp - Emit MLIR Code from OpenMP Statements -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OpenMP Stmt nodes as MLIR code.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTFwd.h"
#include "clang/AST/StmtIterator.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Basic/OpenMPKinds.h"

#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace cir;
using namespace clang;
using namespace mlir::omp;

static void buildDependences(const OMPExecutableDirective &S,
                             OMPTaskDataTy &Data) {

  // First look for 'omp_all_memory' and add this first.
  bool OmpAllMemory = false;
  if (llvm::any_of(
          S.getClausesOfKind<OMPDependClause>(), [](const OMPDependClause *C) {
            return C->getDependencyKind() == OMPC_DEPEND_outallmemory ||
                   C->getDependencyKind() == OMPC_DEPEND_inoutallmemory;
          })) {
    OmpAllMemory = true;
    // Since both OMPC_DEPEND_outallmemory and OMPC_DEPEND_inoutallmemory are
    // equivalent to the runtime, always use OMPC_DEPEND_outallmemory to
    // simplify.
    OMPTaskDataTy::DependData &DD =
        Data.Dependences.emplace_back(OMPC_DEPEND_outallmemory,
                                      /*IteratorExpr=*/nullptr);
    // Add a nullptr Expr to simplify the codegen in emitDependData.
    DD.DepExprs.push_back(nullptr);
  }
  // Add remaining dependences skipping any 'out' or 'inout' if they are
  // overridden by 'omp_all_memory'.
  for (const auto *C : S.getClausesOfKind<OMPDependClause>()) {
    OpenMPDependClauseKind Kind = C->getDependencyKind();
    if (Kind == OMPC_DEPEND_outallmemory || Kind == OMPC_DEPEND_inoutallmemory)
      continue;
    if (OmpAllMemory && (Kind == OMPC_DEPEND_out || Kind == OMPC_DEPEND_inout))
      continue;
    OMPTaskDataTy::DependData &DD =
        Data.Dependences.emplace_back(C->getDependencyKind(), C->getModifier());
    DD.DepExprs.append(C->varlist_begin(), C->varlist_end());
  }
}

mlir::LogicalResult
CIRGenFunction::buildOMPParallelDirective(const OMPParallelDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  // Create a `omp.parallel` op.
  auto parallelOp = builder.create<ParallelOp>(scopeLoc);
  mlir::Block &block = parallelOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);
  // Create a scope for the OpenMP region.
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the body of the region.
        if (buildStmt(S.getCapturedStmt(OpenMPDirectiveKind::OMPD_parallel)
                          ->getCapturedStmt(),
                      /*useCurrentScope=*/true)
                .failed())
          res = mlir::failure();
      });
  // Add the terminator for `omp.parallel`.
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPTaskwaitDirective(const OMPTaskwaitDirective &S) {
  mlir::LogicalResult res = mlir::success();
  OMPTaskDataTy Data;
  buildDependences(S, Data);
  Data.HasNowaitClause = S.hasClausesOfKind<OMPNowaitClause>();
  CGM.getOpenMPRuntime().emitTaskWaitCall(builder, *this,
                                          getLoc(S.getSourceRange()), Data);
  return res;
}
mlir::LogicalResult
CIRGenFunction::buildOMPTaskyieldDirective(const OMPTaskyieldDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.taskyield operation
  CGM.getOpenMPRuntime().emitTaskyieldCall(builder, *this,
                                           getLoc(S.getSourceRange()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPBarrierDirective(const OMPBarrierDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.barrier operation
  CGM.getOpenMPRuntime().emitBarrierCall(builder, *this,
                                         getLoc(S.getSourceRange()));
  return res;
}