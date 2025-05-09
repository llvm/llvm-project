//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC clause nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
namespace clang {
// Simple type-trait to see if the first template arg is one of the list, so we
// can tell whether to `if-constexpr` a bunch of stuff.
template <typename ToTest, typename T, typename... Tys>
constexpr bool isOneOfTypes =
    std::is_same_v<ToTest, T> || isOneOfTypes<ToTest, Tys...>;
template <typename ToTest, typename T>
constexpr bool isOneOfTypes<ToTest, T> = std::is_same_v<ToTest, T>;

// Holds information for emitting clauses for a combined construct. We
// instantiate the clause emitter with this type so that it can use
// if-constexpr to specially handle these.
template <typename CompOpTy> struct CombinedConstructClauseInfo {
  using ComputeOpTy = CompOpTy;
  ComputeOpTy computeOp;
  mlir::acc::LoopOp loopOp;
};

template <typename ToTest> constexpr bool isCombinedType = false;
template <typename T>
constexpr bool isCombinedType<CombinedConstructClauseInfo<T>> = true;

template <typename OpTy>
class OpenACCClauseCIREmitter final
    : public OpenACCClauseVisitor<OpenACCClauseCIREmitter<OpTy>> {
  // Necessary for combined constructs.
  template <typename FriendOpTy> friend class OpenACCClauseCIREmitter;

  OpTy &operation;
  CIRGen::CIRGenFunction &cgf;
  CIRGen::CIRGenBuilderTy &builder;

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

  mlir::Value createConstantInt(mlir::Location loc, unsigned width,
                                int64_t value) {
    mlir::IntegerType ty = mlir::IntegerType::get(
        &cgf.getMLIRContext(), width,
        mlir::IntegerType::SignednessSemantics::Signless);
    auto constOp = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(ty, value));

    return constOp.getResult();
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

  mlir::acc::GangArgType decodeGangType(OpenACCGangKind gk) {
    switch (gk) {
    case OpenACCGangKind::Num:
      return mlir::acc::GangArgType::Num;
    case OpenACCGangKind::Dim:
      return mlir::acc::GangArgType::Dim;
    case OpenACCGangKind::Static:
      return mlir::acc::GangArgType::Static;
    }
    llvm_unreachable("unknown gang kind");
  }

  template <typename U = void,
            typename = std::enable_if_t<isCombinedType<OpTy>, U>>
  void applyToLoopOp(const OpenACCClause &c) {
    // TODO OpenACC: we have to set the insertion scope here correctly still.
    OpenACCClauseCIREmitter<mlir::acc::LoopOp> loopEmitter{
        operation.loopOp, cgf, builder, dirKind, dirLoc};
    loopEmitter.lastDeviceTypeValues = lastDeviceTypeValues;
    loopEmitter.Visit(&c);
  }

  template <typename U = void,
            typename = std::enable_if_t<isCombinedType<OpTy>, U>>
  void applyToComputeOp(const OpenACCClause &c) {
    // TODO OpenACC: we have to set the insertion scope here correctly still.
    OpenACCClauseCIREmitter<typename OpTy::ComputeOpTy> computeEmitter{
        operation.computeOp, cgf, builder, dirKind, dirLoc};
    computeEmitter.lastDeviceTypeValues = lastDeviceTypeValues;
    computeEmitter.Visit(&c);
  }

public:
  OpenACCClauseCIREmitter(OpTy &operation, CIRGen::CIRGenFunction &cgf,
                          CIRGen::CIRGenBuilderTy &builder,
                          OpenACCDirectiveKind dirKind, SourceLocation dirLoc)
      : operation(operation), cgf(cgf), builder(builder), dirKind(dirKind),
        dirLoc(dirLoc) {}

  void VisitClause(const OpenACCClause &clause) {
    clauseNotImplemented(clause);
  }

  void VisitDefaultClause(const OpenACCDefaultClause &clause) {
    // This type-trait checks if 'op'(the first arg) is one of the mlir::acc
    // operations listed in the rest of the arguments.
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      switch (clause.getDefaultClauseKind()) {
      case OpenACCDefaultClauseKind::None:
        operation.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
        break;
      case OpenACCDefaultClauseKind::Present:
        operation.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);
        break;
      case OpenACCDefaultClauseKind::Invalid:
        break;
      }
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitDefaultClause");
    }
  }

  void VisitDeviceTypeClause(const OpenACCDeviceTypeClause &clause) {
    setLastDeviceTypeClause(clause);

    if constexpr (isOneOfTypes<OpTy, mlir::acc::InitOp,
                               mlir::acc::ShutdownOp>) {
      llvm::for_each(
          clause.getArchitectures(), [this](const DeviceTypeArgument &arg) {
            operation.addDeviceType(builder.getContext(),
                                    decodeDeviceType(arg.getIdentifierInfo()));
          });
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::SetOp>) {
      assert(!operation.getDeviceTypeAttr() && "already have device-type?");
      assert(clause.getArchitectures().size() <= 1);

      if (!clause.getArchitectures().empty())
        operation.setDeviceType(
            decodeDeviceType(clause.getArchitectures()[0].getIdentifierInfo()));
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                                      mlir::acc::SerialOp, mlir::acc::KernelsOp,
                                      mlir::acc::DataOp, mlir::acc::LoopOp>) {
      // Nothing to do here, these constructs don't have any IR for these, as
      // they just modify the other clauses IR.  So setting of
      // `lastDeviceTypeValues` (done above) is all we need.
    } else if constexpr (isCombinedType<OpTy>) {
      // Nothing to do here either, combined constructs are just going to use
      // 'lastDeviceTypeValues' to set the value for the child visitor.
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. update, data, routine constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitNumWorkersClause(const OpenACCNumWorkersClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::KernelsOp>) {
      operation.addNumWorkersOperand(builder.getContext(),
                                     createIntExpr(clause.getIntExpr()),
                                     lastDeviceTypeValues);
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::SerialOp>) {
      llvm_unreachable("num_workers not valid on serial");
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitVectorLengthClause(const OpenACCVectorLengthClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::KernelsOp>) {
      operation.addVectorLengthOperand(builder.getContext(),
                                       createIntExpr(clause.getIntExpr()),
                                       lastDeviceTypeValues);
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::SerialOp>) {
      llvm_unreachable("vector_length not valid on serial");
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitAsyncClause(const OpenACCAsyncClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      if (!clause.hasIntExpr())
        operation.addAsyncOnly(builder.getContext(), lastDeviceTypeValues);
      else
        operation.addAsyncOperand(builder.getContext(),
                                  createIntExpr(clause.getIntExpr()),
                                  lastDeviceTypeValues);
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::WaitOp>) {
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
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp>) {
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
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::InitOp,
                               mlir::acc::ShutdownOp, mlir::acc::SetOp,
                               mlir::acc::DataOp, mlir::acc::WaitOp>) {
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
    if constexpr (isOneOfTypes<OpTy, mlir::acc::InitOp, mlir::acc::ShutdownOp,
                               mlir::acc::SetOp>) {
      operation.getDeviceNumMutable().append(
          createIntExpr(clause.getIntExpr()));
    } else {
      llvm_unreachable(
          "init, shutdown, set, are only valid device_num constructs");
    }
  }

  void VisitNumGangsClause(const OpenACCNumGangsClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::KernelsOp>) {
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
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
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
    if constexpr (isOneOfTypes<OpTy, mlir::acc::SetOp>) {
      operation.getDefaultAsyncMutable().append(
          createIntExpr(clause.getIntExpr()));
    } else {
      llvm_unreachable("set, is only valid device_num constructs");
    }
  }

  void VisitSeqClause(const OpenACCSeqClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      operation.addSeq(builder.getContext(), lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Routine construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitAutoClause(const OpenACCAutoClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      operation.addAuto(builder.getContext(), lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Routine, construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitIndependentClause(const OpenACCIndependentClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      operation.addIndependent(builder.getContext(), lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Routine construct remains.
      return clauseNotImplemented(clause);
    }
  }

  void VisitCollapseClause(const OpenACCCollapseClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      llvm::APInt value =
          clause.getIntExpr()->EvaluateKnownConstInt(cgf.cgm.getASTContext());

      value = value.sextOrTrunc(64);
      operation.setCollapseForDeviceTypes(builder.getContext(),
                                          lastDeviceTypeValues, value);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitTileClause(const OpenACCTileClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      llvm::SmallVector<mlir::Value> values;

      for (const Expr *e : clause.getSizeExprs()) {
        mlir::Location exprLoc = cgf.cgm.getLoc(e->getBeginLoc());

        // We represent the * as -1.  Additionally, this is a constant, so we
        // can always just emit it as 64 bits to avoid having to do any more
        // work to determine signedness or size.
        if (isa<OpenACCAsteriskSizeExpr>(e)) {
          values.push_back(createConstantInt(exprLoc, 64, -1));
        } else {
          llvm::APInt curValue =
              e->EvaluateKnownConstInt(cgf.cgm.getASTContext());
          values.push_back(createConstantInt(
              exprLoc, 64, curValue.sextOrTrunc(64).getSExtValue()));
        }
      }

      operation.setTileForDeviceTypes(builder.getContext(),
                                      lastDeviceTypeValues, values);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitWorkerClause(const OpenACCWorkerClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      if (clause.hasIntExpr())
        operation.addWorkerNumOperand(builder.getContext(),
                                      createIntExpr(clause.getIntExpr()),
                                      lastDeviceTypeValues);
      else
        operation.addEmptyWorker(builder.getContext(), lastDeviceTypeValues);

    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitVectorClause(const OpenACCVectorClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      if (clause.hasIntExpr())
        operation.addVectorOperand(builder.getContext(),
                                   createIntExpr(clause.getIntExpr()),
                                   lastDeviceTypeValues);
      else
        operation.addEmptyVector(builder.getContext(), lastDeviceTypeValues);

    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitGangClause(const OpenACCGangClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      if (clause.getNumExprs() == 0) {
        operation.addEmptyGang(builder.getContext(), lastDeviceTypeValues);
      } else {
        llvm::SmallVector<mlir::Value> values;
        llvm::SmallVector<mlir::acc::GangArgType> argTypes;
        for (unsigned i : llvm::index_range(0u, clause.getNumExprs())) {
          auto [kind, expr] = clause.getExpr(i);
          mlir::Location exprLoc = cgf.cgm.getLoc(expr->getBeginLoc());
          argTypes.push_back(decodeGangType(kind));
          if (kind == OpenACCGangKind::Dim) {
            llvm::APInt curValue =
                expr->EvaluateKnownConstInt(cgf.cgm.getASTContext());
            // The value is 1, 2, or 3, but the type isn't necessarily smaller
            // than 64.
            curValue = curValue.sextOrTrunc(64);
            values.push_back(
                createConstantInt(exprLoc, 64, curValue.getSExtValue()));
          } else if (isa<OpenACCAsteriskSizeExpr>(expr)) {
            values.push_back(createConstantInt(exprLoc, 64, -1));
          } else {
            values.push_back(createIntExpr(expr));
          }
        }

        operation.addGangOperands(builder.getContext(), lastDeviceTypeValues,
                                  argTypes, values);
      }
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain.
      return clauseNotImplemented(clause);
    }
  }
};

template <typename OpTy>
auto makeClauseEmitter(OpTy &op, CIRGen::CIRGenFunction &cgf,
                       CIRGen::CIRGenBuilderTy &builder,
                       OpenACCDirectiveKind dirKind, SourceLocation dirLoc) {
  return OpenACCClauseCIREmitter<OpTy>(op, cgf, builder, dirKind, dirLoc);
}

} // namespace clang
