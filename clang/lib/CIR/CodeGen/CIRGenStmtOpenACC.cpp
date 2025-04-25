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

  const OpenACCDeviceTypeClause *lastDeviceTypeClause = nullptr;

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

  // Overload of this function that only returns the device-types list.
  mlir::ArrayAttr
  handleDeviceTypeAffectedClause(mlir::ArrayAttr existingDeviceTypes) {
    mlir::ValueRange argument;
    mlir::MutableOperandRange range{operation};

    return handleDeviceTypeAffectedClause(existingDeviceTypes, argument, range);
  }
  // Overload of this function for when 'segments' aren't necessary.
  mlir::ArrayAttr
  handleDeviceTypeAffectedClause(mlir::ArrayAttr existingDeviceTypes,
                                 mlir::ValueRange argument,
                                 mlir::MutableOperandRange argCollection) {
    llvm::SmallVector<int32_t> segments;
    assert(argument.size() <= 1 &&
           "Overload only for cases where segments don't need to be added");
    return handleDeviceTypeAffectedClause(existingDeviceTypes, argument,
                                          argCollection, segments);
  }

  // Handle a clause affected by the 'device_type' to the point that they need
  // to have attributes added in the correct/corresponding order, such as
  // 'num_workers' or 'vector_length' on a compute construct. The 'argument' is
  // a collection of operands that need to be appended to the `argCollection` as
  // we're adding a 'device_type' entry.  If there is more than 0 elements in
  // the 'argument', the collection must be non-null, as it is needed to add to
  // it.
  // As some clauses, such as 'num_gangs' or 'wait' require a 'segments' list to
  // be maintained, this takes a list of segments that will be updated with the
  // proper counts as 'argument' elements are added.
  //
  // In MLIR, the 'operands' are stored as a large array, with a separate array
  // of 'segments' that show which 'operand' applies to which 'operand-kind'.
  // That is, a 'num_workers' operand-kind or 'num_vectors' operand-kind.
  //
  // So the operands array might have 4 elements, but the 'segments' array will
  // be something like:
  //
  // {0, 0, 0, 2, 0, 1, 1, 0, 0...}
  //
  // Where each position belongs to a specific 'operand-kind'.  So that
  // specifies that whichever operand-kind corresponds with index '3' has 2
  // elements, and should take the 1st 2 operands off the list (since all
  // preceding values are 0). operand-kinds corresponding to 5 and 6 each have
  // 1 element.
  //
  // Fortunately, the `MutableOperandRange` append function actually takes care
  // of that for us at the 'top level'.
  //
  // However, in cases like `num_gangs' or 'wait', where each individual
  // 'element' might be itself array-like, there is a separate 'segments' array
  // for them. So in the case of:
  //
  // device_type(nvidia, radeon) num_gangs(1, 2, 3)
  //
  // We have to emit that as TWO arrays into the IR (where the device_type is an
  // attribute), so they look like:
  //
  // num_gangs({One : i32, Two : i32, Three : i32} [#acc.device_type<nvidia>],\
  //           {One : i32, Two : i32, Three : i32} [#acc.device_type<radeon>])
  //
  // When stored in the 'operands' list, the top-level 'segment' for
  // 'num_gangs' just shows 6 elements. In order to get the array-like
  // apperance, the 'numGangsSegments' list is kept as well. In the above case,
  // we've inserted 6 operands, so the 'numGangsSegments' must contain 2
  // elements, 1 per array, and each will have a value of 3.  The verifier will
  // ensure that the collections counts are correct.
  mlir::ArrayAttr
  handleDeviceTypeAffectedClause(mlir::ArrayAttr existingDeviceTypes,
                                 mlir::ValueRange argument,
                                 mlir::MutableOperandRange argCollection,
                                 llvm::SmallVector<int32_t> &segments) {
    llvm::SmallVector<mlir::Attribute> deviceTypes;

    // Collect the 'existing' device-type attributes so we can re-create them
    // and insert them.
    if (existingDeviceTypes) {
      for (const mlir::Attribute &Attr : existingDeviceTypes)
        deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
            builder.getContext(),
            cast<mlir::acc::DeviceTypeAttr>(Attr).getValue()));
    }

    // Insert 1 version of the 'expr' to the NumWorkers list per-current
    // device type.
    if (lastDeviceTypeClause) {
      for (const DeviceTypeArgument &arch :
           lastDeviceTypeClause->getArchitectures()) {
        deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
            builder.getContext(), decodeDeviceType(arch.getIdentifierInfo())));
        if (!argument.empty()) {
          argCollection.append(argument);
          segments.push_back(argument.size());
        }
      }
    } else {
      // Else, we just add a single for 'none'.
      deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
          builder.getContext(), mlir::acc::DeviceType::None));
      if (!argument.empty()) {
        argCollection.append(argument);
        segments.push_back(argument.size());
      }
    }

    return mlir::ArrayAttr::get(builder.getContext(), deviceTypes);
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
    lastDeviceTypeClause = &clause;
    if constexpr (isOneOfTypes<OpTy, InitOp, ShutdownOp>) {
      llvm::SmallVector<mlir::Attribute> deviceTypes;
      std::optional<mlir::ArrayAttr> existingDeviceTypes =
          operation.getDeviceTypes();

      // Ensure we keep the existing ones, and in the correct 'new' order.
      if (existingDeviceTypes) {
        for (mlir::Attribute attr : *existingDeviceTypes)
          deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
              builder.getContext(),
              cast<mlir::acc::DeviceTypeAttr>(attr).getValue()));
      }

      for (const DeviceTypeArgument &arg : clause.getArchitectures()) {
        deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
            builder.getContext(), decodeDeviceType(arg.getIdentifierInfo())));
      }
      operation.removeDeviceTypesAttr();
      operation.setDeviceTypesAttr(
          mlir::ArrayAttr::get(builder.getContext(), deviceTypes));
    } else if constexpr (isOneOfTypes<OpTy, SetOp>) {
      assert(!operation.getDeviceTypeAttr() && "already have device-type?");
      assert(clause.getArchitectures().size() <= 1);

      if (!clause.getArchitectures().empty())
        operation.setDeviceType(
            decodeDeviceType(clause.getArchitectures()[0].getIdentifierInfo()));
    } else if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp,
                                      DataOp>) {
      // Nothing to do here, these constructs don't have any IR for these, as
      // they just modify the other clauses IR.  So setting of `lastDeviceType`
      // (done above) is all we need.
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. update, data, loop, routine, combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitNumWorkersClause(const OpenACCNumWorkersClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, KernelsOp>) {
      mlir::MutableOperandRange range = operation.getNumWorkersMutable();
      operation.setNumWorkersDeviceTypeAttr(handleDeviceTypeAffectedClause(
          operation.getNumWorkersDeviceTypeAttr(),
          createIntExpr(clause.getIntExpr()), range));
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
      mlir::MutableOperandRange range = operation.getVectorLengthMutable();
      operation.setVectorLengthDeviceTypeAttr(handleDeviceTypeAffectedClause(
          operation.getVectorLengthDeviceTypeAttr(),
          createIntExpr(clause.getIntExpr()), range));
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
      if (!clause.hasIntExpr()) {
        operation.setAsyncOnlyAttr(
            handleDeviceTypeAffectedClause(operation.getAsyncOnlyAttr()));
      } else {
        mlir::MutableOperandRange range = operation.getAsyncOperandsMutable();
        operation.setAsyncOperandsDeviceTypeAttr(handleDeviceTypeAffectedClause(
            operation.getAsyncOperandsDeviceTypeAttr(),
            createIntExpr(clause.getIntExpr()), range));
      }
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

      llvm::SmallVector<int32_t> segments;
      if (operation.getNumGangsSegments())
        llvm::copy(*operation.getNumGangsSegments(),
                   std::back_inserter(segments));

      mlir::MutableOperandRange range = operation.getNumGangsMutable();
      operation.setNumGangsDeviceTypeAttr(handleDeviceTypeAffectedClause(
          operation.getNumGangsDeviceTypeAttr(), values, range, segments));
      operation.setNumGangsSegments(llvm::ArrayRef<int32_t>{segments});
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
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

    for (Expr *QueueExpr  : s.getQueueIdExprs())
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
