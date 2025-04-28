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
#include <type_traits>

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
// Simple type-trait to see if the first template arg is one of the list, so we
// can tell whether to `if-constexpr` a bunch of stuff.
template <typename ToTest, typename T, typename... Tys>
constexpr bool isOneOfTypes =
    std::is_same_v<ToTest, T> || isOneOfTypes<ToTest, Tys...>;
template <typename ToTest, typename T>
constexpr bool isOneOfTypes<ToTest, T> = std::is_same_v<ToTest, T>;

template <typename OpTy>
class OpenACCClauseCIREmitter final
    : public OpenACCClauseVisitor<OpenACCClauseCIREmitter<OpTy>> {
  OpTy &operation;
  CIRGenFunction &cgf;
  CIRGenBuilderTy &builder;

  // This is necessary since a few of the clauses emit differently based on the
  // directive kind they are attached to.
  OpenACCDirectiveKind dirKind;
  // TODO(cir): This source location should be able to go away once the NYI
  // diagnostics are gone.
  SourceLocation dirLoc;

  llvm::SmallVector<mlir::acc::DeviceType> lastDeviceTypeValues;

  void setLastDeviceTypeClause(const OpenACCDeviceTypeClause &clause) {
    lastDeviceTypeValues.clear();

    llvm::for_each(clause.getArchitectures(),
                   [this](const DeviceTypeArgument &arg) {
                     lastDeviceTypeValues.push_back(
                         decodeDeviceType(arg.getIdentifierInfo()));
                   });
  }

  void clauseNotImplemented(const OpenACCClause &c) {
    cgf.cgm.errorNYI(c.getSourceRange(), "OpenACC Clause", c.getClauseKind());
  }

  mlir::Value createIntExpr(const Expr *intExpr) {
    mlir::Value expr = cgf.emitScalarExpr(intExpr);
    mlir::Location exprLoc = cgf.cgm.getLoc(intExpr->getBeginLoc());

    mlir::IntegerType targetType = mlir::IntegerType::get(
        &cgf.getMLIRContext(), cgf.getContext().getIntWidth(intExpr->getType()),
        intExpr->getType()->isSignedIntegerOrEnumerationType()
            ? mlir::IntegerType::SignednessSemantics::Signed
            : mlir::IntegerType::SignednessSemantics::Unsigned);

    auto conversionOp = builder.create<mlir::UnrealizedConversionCastOp>(
        exprLoc, targetType, expr);
    return conversionOp.getResult(0);
  }

  // 'condition' as an OpenACC grammar production is used for 'if' and (some
  // variants of) 'self'.  It needs to be emitted as a signless-1-bit value, so
  // this function emits the expression, then sets the unrealized conversion
  // cast correctly, and returns the completed value.
  mlir::Value createCondition(const Expr *condExpr) {
    mlir::Value condition = cgf.evaluateExprAsBool(condExpr);
    mlir::Location exprLoc = cgf.cgm.getLoc(condExpr->getBeginLoc());
    mlir::IntegerType targetType = mlir::IntegerType::get(
        &cgf.getMLIRContext(), /*width=*/1,
        mlir::IntegerType::SignednessSemantics::Signless);
    auto conversionOp = builder.create<mlir::UnrealizedConversionCastOp>(
        exprLoc, targetType, condition);
    return conversionOp.getResult(0);
  }

  mlir::acc::DeviceType decodeDeviceType(const IdentifierInfo *ii) {
    // '*' case leaves no identifier-info, just a nullptr.
    if (!ii)
      return mlir::acc::DeviceType::Star;
    return llvm::StringSwitch<mlir::acc::DeviceType>(ii->getName())
        .CaseLower("default", mlir::acc::DeviceType::Default)
        .CaseLower("host", mlir::acc::DeviceType::Host)
        .CaseLower("multicore", mlir::acc::DeviceType::Multicore)
        .CasesLower("nvidia", "acc_device_nvidia",
                    mlir::acc::DeviceType::Nvidia)
        .CaseLower("radeon", mlir::acc::DeviceType::Radeon);
  }

public:
  OpenACCClauseCIREmitter(OpTy &operation, CIRGenFunction &cgf,
                          CIRGenBuilderTy &builder,
                          OpenACCDirectiveKind dirKind, SourceLocation dirLoc)
      : operation(operation), cgf(cgf), builder(builder), dirKind(dirKind),
        dirLoc(dirLoc) {}

  void VisitClause(const OpenACCClause &clause) {
    clauseNotImplemented(clause);
  }

  void VisitDefaultClause(const OpenACCDefaultClause &clause) {
    // This type-trait checks if 'op'(the first arg) is one of the mlir::acc
    // operations listed in the rest of the arguments.
    if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp, DataOp>) {
      switch (clause.getDefaultClauseKind()) {
      case OpenACCDefaultClauseKind::None:
        operation.setDefaultAttr(ClauseDefaultValue::None);
        break;
      case OpenACCDefaultClauseKind::Present:
        operation.setDefaultAttr(ClauseDefaultValue::Present);
        break;
      case OpenACCDefaultClauseKind::Invalid:
        break;
      }
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitDeviceTypeClause(const OpenACCDeviceTypeClause &clause) {
    setLastDeviceTypeClause(clause);

    if constexpr (isOneOfTypes<OpTy, InitOp, ShutdownOp>) {
      llvm::for_each(
          clause.getArchitectures(), [this](const DeviceTypeArgument &arg) {
            operation.addDeviceType(builder.getContext(),
                                    decodeDeviceType(arg.getIdentifierInfo()));
          });
    } else if constexpr (isOneOfTypes<OpTy, SetOp>) {
      assert(!operation.getDeviceTypeAttr() && "already have device-type?");
      assert(clause.getArchitectures().size() <= 1);

      if (!clause.getArchitectures().empty())
        operation.setDeviceType(
            decodeDeviceType(clause.getArchitectures()[0].getIdentifierInfo()));
    } else if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp,
                                      DataOp>) {
      // Nothing to do here, these constructs don't have any IR for these, as
      // they just modify the other clauses IR.  So setting of
      // `lastDeviceTypeValues` (done above) is all we need.
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. update, data, loop, routine, combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitNumWorkersClause(const OpenACCNumWorkersClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, KernelsOp>) {
      operation.addNumWorkersOperand(builder.getContext(),
                                     createIntExpr(clause.getIntExpr()),
                                     lastDeviceTypeValues);
    } else if constexpr (isOneOfTypes<OpTy, SerialOp>) {
      llvm_unreachable("num_workers not valid on serial");
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitVectorLengthClause(const OpenACCVectorLengthClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, KernelsOp>) {
      operation.addVectorLengthOperand(builder.getContext(),
                                       createIntExpr(clause.getIntExpr()),
                                       lastDeviceTypeValues);
    } else if constexpr (isOneOfTypes<OpTy, SerialOp>) {
      llvm_unreachable("vector_length not valid on serial");
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitAsyncClause(const OpenACCAsyncClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp, DataOp>) {
      if (!clause.hasIntExpr())
        operation.addAsyncOnly(builder.getContext(), lastDeviceTypeValues);
      else
        operation.addAsyncOperand(builder.getContext(),
                                  createIntExpr(clause.getIntExpr()),
                                  lastDeviceTypeValues);
    } else if constexpr (isOneOfTypes<OpTy, WaitOp>) {
      // Wait doesn't have a device_type, so its handling here is slightly
      // different.
      if (!clause.hasIntExpr())
        operation.setAsync(true);
      else
        operation.getAsyncOperandMutable().append(
            createIntExpr(clause.getIntExpr()));
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain. Data, enter data, exit data,
      // update, combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitSelfClause(const OpenACCSelfClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp>) {
      if (clause.isEmptySelfClause()) {
        operation.setSelfAttr(true);
      } else if (clause.isConditionExprClause()) {
        assert(clause.hasConditionExpr());
        operation.getSelfCondMutable().append(
            createCondition(clause.getConditionExpr()));
      } else {
        llvm_unreachable("var-list version of self shouldn't get here");
      }
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. If, combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitIfClause(const OpenACCIfClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp, InitOp,
                               ShutdownOp, SetOp, DataOp, WaitOp>) {
      operation.getIfCondMutable().append(
          createCondition(clause.getConditionExpr()));
    } else {
      // 'if' applies to most of the constructs, but hold off on lowering them
      // until we can write tests/know what we're doing with codegen to make
      // sure we get it right.
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Enter data, exit data, host_data, update, combined 
      // constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitDeviceNumClause(const OpenACCDeviceNumClause &clause) {
    if constexpr (isOneOfTypes<OpTy, InitOp, ShutdownOp, SetOp>) {
      operation.getDeviceNumMutable().append(
          createIntExpr(clause.getIntExpr()));
    } else {
      llvm_unreachable(
          "init, shutdown, set, are only valid device_num constructs");
    }
  }

  void VisitNumGangsClause(const OpenACCNumGangsClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, KernelsOp>) {
      llvm::SmallVector<mlir::Value> values;
      for (const Expr *E : clause.getIntExprs())
        values.push_back(createIntExpr(E));

      operation.addNumGangsOperands(builder.getContext(), values,
                                    lastDeviceTypeValues);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitWaitClause(const OpenACCWaitClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp, DataOp>) {
      if (!clause.hasExprs()) {
        operation.addWaitOnly(builder.getContext(), lastDeviceTypeValues);
      } else {
        llvm::SmallVector<mlir::Value> values;
        if (clause.hasDevNumExpr())
          values.push_back(createIntExpr(clause.getDevNumExpr()));
        for (const Expr *E : clause.getQueueIdExprs())
          values.push_back(createIntExpr(E));
        operation.addWaitOperands(builder.getContext(), clause.hasDevNumExpr(),
                                  values, lastDeviceTypeValues);
      }
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Enter data, exit data, update, Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitDefaultAsyncClause(const OpenACCDefaultAsyncClause &clause) {
    if constexpr (isOneOfTypes<OpTy, SetOp>) {
      operation.getDefaultAsyncMutable().append(
          createIntExpr(clause.getIntExpr()));
    } else {
      llvm_unreachable("set, is only valid device_num constructs");
    }
  }
};

template <typename OpTy>
auto makeClauseEmitter(OpTy &op, CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                       OpenACCDirectiveKind dirKind, SourceLocation dirLoc) {
  return OpenACCClauseCIREmitter<OpTy>(op, cgf, builder, dirKind, dirLoc);
}

} // namespace

template <typename Op, typename TermOp>
mlir::LogicalResult CIRGenFunction::emitOpenACCOpAssociatedStmt(
    mlir::Location start, mlir::Location end, OpenACCDirectiveKind dirKind,
    SourceLocation dirLoc, llvm::ArrayRef<const OpenACCClause *> clauses,
    const Stmt *associatedStmt) {
  mlir::LogicalResult res = mlir::success();

  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto op = builder.create<Op>(start, retTy, operands);

  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    // Sets insertion point before the 'op', since every new expression needs to
    // be before the operation.
    builder.setInsertionPoint(op);
    makeClauseEmitter(op, *this, builder, dirKind, dirLoc)
        .VisitClauseList(clauses);
  }

  {
    mlir::Block &block = op.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, start, builder.getInsertionBlock()};
    res = emitStmt(associatedStmt, /*useCurrentScope=*/true);

    builder.create<TermOp>(end);
  }
  return res;
}

template <typename Op>
Op CIRGenFunction::emitOpenACCOp(
    mlir::Location start, OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
    llvm::ArrayRef<const OpenACCClause *> clauses) {
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto op = builder.create<Op>(start, retTy, operands);

  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    // Sets insertion point before the 'op', since every new expression needs to
    // be before the operation.
    builder.setInsertionPoint(op);
    makeClauseEmitter(op, *this, builder, dirKind, dirLoc)
        .VisitClauseList(clauses);
  }
  return op;
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCComputeConstruct(const OpenACCComputeConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  switch (s.getDirectiveKind()) {
  case OpenACCDirectiveKind::Parallel:
    return emitOpenACCOpAssociatedStmt<ParallelOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  case OpenACCDirectiveKind::Serial:
    return emitOpenACCOpAssociatedStmt<SerialOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  case OpenACCDirectiveKind::Kernels:
    return emitOpenACCOpAssociatedStmt<KernelsOp, mlir::acc::TerminatorOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  default:
    llvm_unreachable("invalid compute construct kind");
  }
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCDataConstruct(const OpenACCDataConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  return emitOpenACCOpAssociatedStmt<DataOp, mlir::acc::TerminatorOp>(
      start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
      s.getStructuredBlock());
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCInitConstruct(const OpenACCInitConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<InitOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                               s.clauses());
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCSetConstruct(const OpenACCSetConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<SetOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                              s.clauses());
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitOpenACCShutdownConstruct(
    const OpenACCShutdownConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<ShutdownOp>(start, s.getDirectiveKind(),
                                   s.getDirectiveLoc(), s.clauses());
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  auto waitOp = emitOpenACCOp<WaitOp>(start, s.getDirectiveKind(),
                                   s.getDirectiveLoc(), s.clauses());

  auto createIntExpr = [this](const Expr *intExpr) {
    mlir::Value expr = emitScalarExpr(intExpr);
    mlir::Location exprLoc = cgm.getLoc(intExpr->getBeginLoc());

    mlir::IntegerType targetType = mlir::IntegerType::get(
        &getMLIRContext(), getContext().getIntWidth(intExpr->getType()),
        intExpr->getType()->isSignedIntegerOrEnumerationType()
            ? mlir::IntegerType::SignednessSemantics::Signed
            : mlir::IntegerType::SignednessSemantics::Unsigned);

    auto conversionOp = builder.create<mlir::UnrealizedConversionCastOp>(
        exprLoc, targetType, expr);
    return conversionOp.getResult(0);
  };

  // Emit the correct 'wait' clauses.
  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPoint(waitOp);

    if (s.hasDevNumExpr())
      waitOp.getWaitDevnumMutable().append(createIntExpr(s.getDevNumExpr()));

    for (Expr *QueueExpr : s.getQueueIdExprs())
      waitOp.getWaitOperandsMutable().append(createIntExpr(QueueExpr));
  }

  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCLoopConstruct(const OpenACCLoopConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Loop Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCCombinedConstruct(
    const OpenACCCombinedConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Combined Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCEnterDataConstruct(
    const OpenACCEnterDataConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC EnterData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCExitDataConstruct(
    const OpenACCExitDataConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC ExitData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCHostDataConstruct(
    const OpenACCHostDataConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC HostData Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Update Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Atomic Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Cache Construct");
  return mlir::failure();
}
