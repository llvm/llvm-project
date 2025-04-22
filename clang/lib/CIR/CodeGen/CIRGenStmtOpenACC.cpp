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
      return clauseNotImplemented(clause);
    }
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
    } else if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp>) {
      // Nothing to do here, these constructs don't have any IR for these, as
      // they just modify the other clauses IR.  So setting of `lastDeviceType`
      // (done above) is all we need.
    } else {
      return clauseNotImplemented(clause);
    }
  }

  void VisitNumWorkersClause(const OpenACCNumWorkersClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, KernelsOp>) {
      // Collect the 'existing' device-type attributes so we can re-create them
      // and insert them.
      llvm::SmallVector<mlir::Attribute> deviceTypes;
      mlir::ArrayAttr existingDeviceTypes =
          operation.getNumWorkersDeviceTypeAttr();

      if (existingDeviceTypes) {
        for (mlir::Attribute Attr : existingDeviceTypes)
          deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
              builder.getContext(),
              cast<mlir::acc::DeviceTypeAttr>(Attr).getValue()));
      }

      // Insert 1 version of the 'int-expr' to the NumWorkers list per-current
      // device type.
      mlir::Value intExpr = createIntExpr(clause.getIntExpr());
      if (lastDeviceTypeClause) {
        for (const DeviceTypeArgument &arg :
             lastDeviceTypeClause->getArchitectures()) {
          deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
              builder.getContext(), decodeDeviceType(arg.getIdentifierInfo())));
          operation.getNumWorkersMutable().append(intExpr);
        }
      } else {
        // Else, we just add a single for 'none'.
        deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
            builder.getContext(), mlir::acc::DeviceType::None));
        operation.getNumWorkersMutable().append(intExpr);
      }

      operation.setNumWorkersDeviceTypeAttr(
          mlir::ArrayAttr::get(builder.getContext(), deviceTypes));
    } else if constexpr (isOneOfTypes<OpTy, SerialOp>) {
      llvm_unreachable("num_workers not valid on serial");
    } else {
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
      return clauseNotImplemented(clause);
    }
  }

  void VisitIfClause(const OpenACCIfClause &clause) {
    if constexpr (isOneOfTypes<OpTy, ParallelOp, SerialOp, KernelsOp>) {
      operation.getIfCondMutable().append(
          createCondition(clause.getConditionExpr()));
    } else {
      // 'if' applies to most of the constructs, but hold off on lowering them
      // until we can write tests/know what we're doing with codegen to make
      // sure we get it right.
      return clauseNotImplemented(clause);
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
mlir::LogicalResult CIRGenFunction::emitOpenACCOp(
    mlir::Location start, OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
    llvm::ArrayRef<const OpenACCClause *> clauses) {
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
  return res;
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
  return emitOpenACCOp<InitOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                               s.clauses());
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCSetConstruct(const OpenACCSetConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  return emitOpenACCOp<SetOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                              s.clauses());
}

mlir::LogicalResult CIRGenFunction::emitOpenACCShutdownConstruct(
    const OpenACCShutdownConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  return emitOpenACCOp<ShutdownOp>(start, s.getDirectiveKind(),
                                   s.getDirectiveLoc(), s.clauses());
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
CIRGenFunction::emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Wait Construct");
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
