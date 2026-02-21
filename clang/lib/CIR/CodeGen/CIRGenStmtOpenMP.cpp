//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenMP Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace clang;
using namespace clang::CIRGen;

mlir::LogicalResult
CIRGenFunction::emitOMPScopeDirective(const OMPScopeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScopeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPErrorDirective(const OMPErrorDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPErrorDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPParallelDirective(const OMPParallelDirective &s) {
  mlir::LogicalResult res = mlir::success();
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  auto parallelOp =
      mlir::omp::ParallelOp::create(builder, begin, retTy, operands);
  emitOpenMPClauses(parallelOp, s.clauses());

  {
    mlir::Block &block = parallelOp.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    if (s.hasCancel())
      getCIRGenModule().errorNYI(s.getBeginLoc(),
                                 "OpenMP Parallel with Cancel");
    if (s.getTaskReductionRefExpr())
      getCIRGenModule().errorNYI(s.getBeginLoc(),
                                 "OpenMP Parallel with Task Reduction");
    // Don't lower the captured statement directly since this will be
    // special-cased depending on the kind of OpenMP directive that is the
    // parent, also the non-OpenMP context captured statements lowering does
    // not apply directly.
    const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_parallel);
    const Stmt *bodyStmt = cs->getCapturedStmt();
    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);
    mlir::omp::TerminatorOp::create(builder, end);
  }
  return res;
}

// Helpers and implementation for emitOMPForDirective, which lowers an
// OMPForDirective into an omp.wsloop + omp.loop_nest.

namespace {
/// Extract integer literal value from an expression, if present.
static std::optional<int64_t> getIntLiteralValue(const Expr *expr) {
  if (const auto *intLit = dyn_cast<IntegerLiteral>(expr->IgnoreImpCasts()))
    return intLit->getValue().getSExtValue();
  return std::nullopt;
}

/// Ensure a CIR value has the given CIR integer type, inserting an integral
/// cast if necessary. Loads through CIR pointers first.
static mlir::Value ensureCIRIntType(CIRGenBuilderTy &builder,
                                    mlir::Location loc, mlir::Value cirValue,
                                    cir::IntType targetCIRType) {
  if (mlir::isa<cir::PointerType>(cirValue.getType()))
    cirValue = cir::LoadOp::create(builder, loc, cirValue).getResult();

  if (cirValue.getType() == targetCIRType)
    return cirValue;

  return builder.createCast(loc, cir::CastKind::integral, cirValue,
                            targetCIRType);
}

/// Convert a CIR integer value to a standard MLIR integer type suitable for
/// use as an omp.loop_nest operand.
static mlir::Value cirIntToStdInt(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value cirValue) {
  auto cirIntType = mlir::cast<cir::IntType>(cirValue.getType());
  mlir::Type stdIntType = builder.getIntegerType(cirIntType.getWidth());
  return mlir::UnrealizedConversionCastOp::create(builder, loc, stdIntType,
                                                  cirValue)
      .getResult(0);
}
} // anonymous namespace

mlir::LogicalResult
CIRGenFunction::emitOMPForDirective(const OMPForDirective &s) {

  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());

  // Extract the underlying canonical `for` loop from the CapturedStmt
  const CapturedStmt *capturedStmt = s.getInnermostCapturedStmt();
  const ForStmt *forStmt = dyn_cast<ForStmt>(capturedStmt->getCapturedStmt());

  if (!forStmt) {
    return mlir::failure();
  }

  // Loop bounds are first built as CIR integer values, then converted to
  // standard MLIR integers via UnrealizedConversionCastOp before being
  // passed to omp.loop_nest (which requires IntLikeType operands).
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool inclusive = false;

  // Extract loop variable type and lower bound.
  const auto *declStmt = dyn_cast_or_null<DeclStmt>(forStmt->getInit());
  const auto *varDecl =
      declStmt ? dyn_cast<VarDecl>(declStmt->getSingleDecl()) : nullptr;

  if (!varDecl)
    return mlir::failure();

  // The loop variable's CIR integer type is the canonical type for all bounds.
  QualType loopVarQType = varDecl->getType();
  auto cirType = convertType(loopVarQType);
  auto cirIntType = mlir::cast<cir::IntType>(cirType);

  // Extract lower bound.
  if (!varDecl->hasInit())
    return mlir::failure();

  if (auto constVal = getIntLiteralValue(varDecl->getInit())) {
    lowerBound = builder.getConstInt(begin, cirIntType, *constVal);
  } else {
    mlir::Value cirValue = emitScalarExpr(varDecl->getInit());
    lowerBound = ensureCIRIntType(builder, begin, cirValue, cirIntType);
  }

  // Extract upper bound and comparison operator.
  const auto *condBinOp = dyn_cast_or_null<BinaryOperator>(forStmt->getCond());
  if (!condBinOp)
    return mlir::failure();

  BinaryOperatorKind opKind = condBinOp->getOpcode();

  // Determine which side of the comparison holds the upper bound.
  // Canonical forms: `i < ub`, `i <= ub` (var on LHS, bound on RHS)
  //                  `ub > i`, `ub >= i` (bound on LHS, var on RHS)
  const Expr *boundExpr = nullptr;
  if (opKind == BO_LT || opKind == BO_LE) {
    boundExpr = condBinOp->getRHS();
    inclusive = (opKind == BO_LE);
  } else if (opKind == BO_GT || opKind == BO_GE) {
    boundExpr = condBinOp->getLHS();
    inclusive = (opKind == BO_GE);
  } else {
    return mlir::failure();
  }

  if (auto constVal = getIntLiteralValue(boundExpr)) {
    upperBound = builder.getConstInt(begin, cirIntType, *constVal);
  } else {
    mlir::Value cirValue = emitScalarExpr(boundExpr);
    upperBound = ensureCIRIntType(builder, begin, cirValue, cirIntType);
  }

  // Extract step.
  if (const auto *unaryOp =
          dyn_cast_or_null<UnaryOperator>(forStmt->getInc())) {
    int64_t val = unaryOp->isIncrementOp() ? 1 : -1;
    step = builder.getConstInt(begin, cirIntType, val);
  } else if (const auto *binOp =
                 dyn_cast_or_null<BinaryOperator>(forStmt->getInc())) {
    const Expr *stepExpr = nullptr;

    if (binOp->isCompoundAssignmentOp()) {
      stepExpr = binOp->getRHS();
    } else if (binOp->isAssignmentOp()) {
      // i = i + step or i = step + i
      if (auto *subBinOp =
              dyn_cast<BinaryOperator>(binOp->getRHS()->IgnoreImpCasts())) {
        const Expr *lhs = subBinOp->getLHS()->IgnoreImpCasts();
        const Expr *rhs = subBinOp->getRHS()->IgnoreImpCasts();
        // Identify which operand is the loop variable and which is the step.
        if (auto *lhsRef = dyn_cast<DeclRefExpr>(lhs)) {
          stepExpr = (lhsRef->getDecl() == varDecl) ? rhs : lhs;
        } else if (auto *rhsRef = dyn_cast<DeclRefExpr>(rhs)) {
          stepExpr = (rhsRef->getDecl() == varDecl) ? lhs : rhs;
        }
      }
    }

    if (stepExpr) {
      if (auto constVal = getIntLiteralValue(stepExpr)) {
        step = builder.getConstInt(begin, cirIntType, *constVal);
      } else {
        mlir::Value cirValue = emitScalarExpr(stepExpr);
        step = ensureCIRIntType(builder, begin, cirValue, cirIntType);
      }
    }
  }

  // Default to unit step if not recognized.
  if (!step)
    step = builder.getConstInt(begin, cirIntType, 1);

  // Emit init, convert bounds to std integers, and create the wsloop.

  // Emit the loop init statement (e.g. `int i = 0`) to create the alloca
  // for the induction variable *before* the wsloop.
  if (forStmt->getInit())
    if (emitStmt(forStmt->getInit(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

  // Convert CIR integer bounds to standard MLIR integers at the boundary.
  // omp.loop_nest requires IntLikeType (AnyInteger | Index), not CIR types.
  mlir::Value stdLB = cirIntToStdInt(builder, begin, lowerBound);
  mlir::Value stdUB = cirIntToStdInt(builder, begin, upperBound);
  mlir::Value stdStep = cirIntToStdInt(builder, begin, step);
  mlir::Type loopBoundsType = stdLB.getType();

  currentOMPLoopBounds =
      LoopBounds{stdLB, stdUB, stdStep, loopBoundsType, varDecl, inclusive};

  // Create wsloop with empty region
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto wsloopOp = mlir::omp::WsloopOp::create(builder, begin, retTy, operands);

  mlir::Region &region = wsloopOp.getRegion();
  mlir::Block *block = new mlir::Block();
  region.push_back(block);

  // Emit the ForStmt body (will create loop_nest when it detects OpenMP
  // context)
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  if (emitStmt(forStmt, /*useCurrentScope=*/false).failed()) {
    res = mlir::failure();
  }

  // Clear loop-bound state
  currentOMPLoopBounds = std::nullopt;

  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOMPTaskwaitDirective(const OMPTaskwaitDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskwaitDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskyieldDirective(const OMPTaskyieldDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskyieldDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPBarrierDirective(const OMPBarrierDirective &s) {
  mlir::omp::BarrierOp::create(builder, getLoc(s.getBeginLoc()));
  assert(s.clauses().empty() && "omp barrier doesn't support clauses");
  return mlir::success();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMetaDirective(const OMPMetaDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMetaDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCanonicalLoop(const OMPCanonicalLoop &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCanonicalLoop");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSimdDirective(const OMPSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTileDirective(const OMPTileDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTileDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPUnrollDirective(const OMPUnrollDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPUnrollDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFuseDirective(const OMPFuseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFuseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPForSimdDirective(const OMPForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionsDirective(const OMPSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionsDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionDirective(const OMPSectionDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSingleDirective(const OMPSingleDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSingleDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMasterDirective(const OMPMasterDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMasterDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCriticalDirective(const OMPCriticalDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCriticalDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPParallelForDirective(const OMPParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelForSimdDirective(
    const OMPParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterDirective(
    const OMPParallelMasterDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelSectionsDirective(
    const OMPParallelSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelSectionsDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskDirective(const OMPTaskDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskgroupDirective(const OMPTaskgroupDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskgroupDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFlushDirective(const OMPFlushDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFlushDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDepobjDirective(const OMPDepobjDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDepobjDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPScanDirective(const OMPScanDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScanDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPOrderedDirective(const OMPOrderedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPOrderedDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAtomicDirective(const OMPAtomicDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAtomicDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetDirective(const OMPTargetDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTargetDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDirective(const OMPTeamsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTeamsDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPCancellationPointDirective(
    const OMPCancellationPointDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPCancellationPointDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCancelDirective(const OMPCancelDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCancelDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetDataDirective(const OMPTargetDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetEnterDataDirective(
    const OMPTargetEnterDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetEnterDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetExitDataDirective(
    const OMPTargetExitDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetExitDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelDirective(
    const OMPTargetParallelDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForDirective(
    const OMPTargetParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskLoopDirective(const OMPTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTaskLoopSimdDirective(
    const OMPTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopDirective(
    const OMPMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopSimdDirective(
    const OMPMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopDirective(
    const OMPMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopSimdDirective(
    const OMPMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelGenericLoopDirective(
    const OMPParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedDirective(
    const OMPParallelMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopDirective(
    const OMPParallelMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopSimdDirective(
    const OMPParallelMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopDirective(
    const OMPParallelMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopSimdDirective(
    const OMPParallelMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDistributeDirective(const OMPDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForDirective(
    const OMPDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForSimdDirective(
    const OMPDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeSimdDirective(
    const OMPDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelGenericLoopDirective(
    const OMPTargetParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForSimdDirective(
    const OMPTargetParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetSimdDirective(const OMPTargetSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsGenericLoopDirective(
    const OMPTargetTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetUpdateDirective(
    const OMPTargetUpdateDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetUpdateDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeDirective(
    const OMPTeamsDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeSimdDirective(
    const OMPTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDistributeParallelForSimdDirective(
    const OMPTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(), "OpenMP OMPTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeParallelForDirective(
    const OMPTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsGenericLoopDirective(
    const OMPTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDirective(const OMPTargetTeamsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeDirective(
    const OMPTargetTeamsDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForDirective(
    const OMPTargetTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForSimdDirective(
    const OMPTargetTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeSimdDirective(
    const OMPTargetTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInteropDirective(const OMPInteropDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPInteropDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDispatchDirective(const OMPDispatchDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDispatchDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPGenericLoopDirective(const OMPGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPReverseDirective(const OMPReverseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPReverseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInterchangeDirective(const OMPInterchangeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPInterchangeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAssumeDirective(const OMPAssumeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAssumeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMaskedDirective(const OMPMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPStripeDirective(const OMPStripeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPStripeDirective");
  return mlir::failure();
}
