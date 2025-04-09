//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/AST/StmtOpenACC.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace mlir::acc;

namespace {
class OpenACCClauseCIREmitter final
    : public OpenACCClauseVisitor<OpenACCClauseCIREmitter> {
  CIRGenModule &cgm;

  struct AttributeData {
    // Value of the 'default' attribute, added on 'data' and 'compute'/etc
    // constructs as a 'default-attr'.
    std::optional<ClauseDefaultValue> defaultVal = std::nullopt;
  } attrData;

  void clauseNotImplemented(const OpenACCClause &c) {
    cgm.errorNYI(c.getSourceRange(), "OpenACC Clause", c.getClauseKind());
  }

public:
  OpenACCClauseCIREmitter(CIRGenModule &cgm) : cgm(cgm) {}

  void VisitClause(const OpenACCClause &clause) {
    clauseNotImplemented(clause);
  }

  void VisitDefaultClause(const OpenACCDefaultClause &clause) {
    switch (clause.getDefaultClauseKind()) {
    case OpenACCDefaultClauseKind::None:
      attrData.defaultVal = ClauseDefaultValue::None;
      break;
    case OpenACCDefaultClauseKind::Present:
      attrData.defaultVal = ClauseDefaultValue::Present;
      break;
    case OpenACCDefaultClauseKind::Invalid:
      break;
    }
  }

  // Apply any of the clauses that resulted in an 'attribute'.
  template <typename Op> void applyAttributes(Op &op) {
    if (attrData.defaultVal.has_value())
      op.setDefaultAttr(*attrData.defaultVal);
  }
};
} // namespace

template <typename Op, typename TermOp>
mlir::LogicalResult CIRGenFunction::emitOpenACCOpAssociatedStmt(
    mlir::Location start, mlir::Location end,
    llvm::ArrayRef<const OpenACCClause *> clauses, const Stmt *associatedStmt) {
  mlir::LogicalResult res = mlir::success();

  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;

  // Clause-emitter must be here because it might modify operands.
  OpenACCClauseCIREmitter clauseEmitter(getCIRGenModule());
  clauseEmitter.VisitClauseList(clauses);

  auto op = builder.create<Op>(start, retTy, operands);

  // Apply the attributes derived from the clauses.
  clauseEmitter.applyAttributes(op);

  mlir::Block &block = op.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);

  LexicalScope ls{*this, start, builder.getInsertionBlock()};
  res = emitStmt(associatedStmt, /*useCurrentScope=*/true);

  builder.create<TermOp>(end);
  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCComputeConstruct(const OpenACCComputeConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getEnd());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  switch (s.getDirectiveKind()) {
  case OpenACCDirectiveKind::Parallel:
    return emitOpenACCOpAssociatedStmt<ParallelOp, mlir::acc::YieldOp>(
        start, end, s.clauses(), s.getStructuredBlock());
  case OpenACCDirectiveKind::Serial:
    return emitOpenACCOpAssociatedStmt<SerialOp, mlir::acc::YieldOp>(
        start, end, s.clauses(), s.getStructuredBlock());
  case OpenACCDirectiveKind::Kernels:
    return emitOpenACCOpAssociatedStmt<KernelsOp, mlir::acc::TerminatorOp>(
        start, end, s.clauses(), s.getStructuredBlock());
  default:
    llvm_unreachable("invalid compute construct kind");
  }
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCDataConstruct(const OpenACCDataConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getEnd());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  return emitOpenACCOpAssociatedStmt<DataOp, mlir::acc::TerminatorOp>(
      start, end, s.clauses(), s.getStructuredBlock());
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCLoopConstruct(const OpenACCLoopConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Loop Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCCombinedConstruct(
    const OpenACCCombinedConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Combined Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCEnterDataConstruct(
    const OpenACCEnterDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC EnterData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCExitDataConstruct(
    const OpenACCExitDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC ExitData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCHostDataConstruct(
    const OpenACCHostDataConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC HostData Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Wait Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCInitConstruct(const OpenACCInitConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Init Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCShutdownConstruct(
    const OpenACCShutdownConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Shutdown Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCSetConstruct(const OpenACCSetConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Set Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Update Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Atomic Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenACC Cache Construct");
  return mlir::failure();
}
