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

#include "CIRGenFunction.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
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
  // Keep track of the async-clause so that we can shortcut updating the data
  // operands async clauses.
  bool hasAsyncClause = false;
  // Keep track of the data operands so that we can update their async clauses.
  llvm::SmallVector<mlir::Operation *> dataOperands;

  void clauseNotImplemented(const OpenACCClause &c) {
    cgf.cgm.errorNYI(c.getSourceRange(), "OpenACC Clause", c.getClauseKind());
  }

  void setLastDeviceTypeClause(const OpenACCDeviceTypeClause &clause) {
    lastDeviceTypeValues.clear();

    llvm::for_each(clause.getArchitectures(),
                   [this](const DeviceTypeArgument &arg) {
                     lastDeviceTypeValues.push_back(
                         decodeDeviceType(arg.getIdentifierInfo()));
                   });
  }

  mlir::Value emitIntExpr(const Expr *intExpr) {
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

  mlir::Value createConstantInt(SourceLocation loc, unsigned width,
                                int64_t value) {
    return createConstantInt(cgf.cgm.getLoc(loc), width, value);
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
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPoint(operation.loopOp);
    OpenACCClauseCIREmitter<mlir::acc::LoopOp> loopEmitter{
        operation.loopOp, cgf, builder, dirKind, dirLoc};
    loopEmitter.lastDeviceTypeValues = lastDeviceTypeValues;
    loopEmitter.Visit(&c);
  }

  template <typename U = void,
            typename = std::enable_if_t<isCombinedType<OpTy>, U>>
  void applyToComputeOp(const OpenACCClause &c) {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPoint(operation.computeOp);
    OpenACCClauseCIREmitter<typename OpTy::ComputeOpTy> computeEmitter{
        operation.computeOp, cgf, builder, dirKind, dirLoc};

    computeEmitter.lastDeviceTypeValues = lastDeviceTypeValues;

    // Async handler uses the first data operand to figure out where to insert
    // its information if it is present.  This ensures that the new handler will
    // correctly set the insertion point for async.
    if (!dataOperands.empty())
      computeEmitter.dataOperands.push_back(dataOperands.front());
    computeEmitter.Visit(&c);

    // Make sure all of the new data operands are kept track of here. The
    // combined constructs always apply 'async' to only the compute component,
    // so we need to collect these.
    dataOperands.append(computeEmitter.dataOperands);
  }

  struct DataOperandInfo {
    mlir::Location beginLoc;
    mlir::Value varValue;
    llvm::StringRef name;
    llvm::SmallVector<mlir::Value> bounds;
  };

  mlir::Value createBound(mlir::Location boundLoc, mlir::Value lowerBound,
                          mlir::Value upperBound, mlir::Value extent) {
    // Arrays always have a start-idx of 0.
    mlir::Value startIdx = createConstantInt(boundLoc, 64, 0);
    // Stride is always 1 in C/C++.
    mlir::Value stride = createConstantInt(boundLoc, 64, 1);

    auto bound = builder.create<mlir::acc::DataBoundsOp>(boundLoc, lowerBound,
                                                         upperBound);
    bound.getStartIdxMutable().assign(startIdx);
    if (extent)
      bound.getExtentMutable().assign(extent);
    bound.getStrideMutable().assign(stride);

    return bound;
  }

  // A helper function that gets the information from an operand to a data
  // clause, so that it can be used to emit the data operations.
  DataOperandInfo getDataOperandInfo(OpenACCDirectiveKind dk, const Expr *e) {
    // TODO: OpenACC: Cache was different enough as to need a separate
    // `ActOnCacheVar`, so we are going to need to do some investigations here
    // when it comes to implement this for cache.
    if (dk == OpenACCDirectiveKind::Cache) {
      cgf.cgm.errorNYI(e->getSourceRange(),
                       "OpenACC data operand for 'cache' directive");
      return {cgf.cgm.getLoc(e->getBeginLoc()), {}, {}, {}};
    }

    const Expr *curVarExpr = e->IgnoreParenImpCasts();

    mlir::Location exprLoc = cgf.cgm.getLoc(curVarExpr->getBeginLoc());
    llvm::SmallVector<mlir::Value> bounds;

    // Assemble the list of bounds.
    while (isa<ArraySectionExpr, ArraySubscriptExpr>(curVarExpr)) {
      mlir::Location boundLoc = cgf.cgm.getLoc(curVarExpr->getBeginLoc());
      mlir::Value lowerBound;
      mlir::Value upperBound;
      mlir::Value extent;

      if (const auto *section = dyn_cast<ArraySectionExpr>(curVarExpr)) {
        if (const Expr *lb = section->getLowerBound())
          lowerBound = emitIntExpr(lb);
        else
          lowerBound = createConstantInt(boundLoc, 64, 0);

        if (const Expr *len = section->getLength()) {
          extent = emitIntExpr(len);
        } else {
          QualType baseTy = ArraySectionExpr::getBaseOriginalType(
              section->getBase()->IgnoreParenImpCasts());
          // We know this is the case as implicit lengths are only allowed for
          // array types with a constant size, or a dependent size.  AND since
          // we are codegen we know we're not dependent.
          auto *arrayTy = cgf.getContext().getAsConstantArrayType(baseTy);
          // Rather than trying to calculate the extent based on the
          // lower-bound, we can just emit this as an upper bound.
          upperBound =
              createConstantInt(boundLoc, 64, arrayTy->getLimitedSize() - 1);
        }

        curVarExpr = section->getBase()->IgnoreParenImpCasts();
      } else {
        const auto *subscript = cast<ArraySubscriptExpr>(curVarExpr);

        lowerBound = emitIntExpr(subscript->getIdx());
        // Length of an array index is always 1.
        extent = createConstantInt(boundLoc, 64, 1);
        curVarExpr = subscript->getBase()->IgnoreParenImpCasts();
      }

      bounds.push_back(createBound(boundLoc, lowerBound, upperBound, extent));
    }

    // TODO: OpenACC: if this is a member expr, emit the VarPtrPtr correctly.
    if (isa<MemberExpr>(curVarExpr)) {
      cgf.cgm.errorNYI(curVarExpr->getSourceRange(),
                       "OpenACC Data clause member expr");
      return {exprLoc, {}, {}, std::move(bounds)};
    }

    // Sema has made sure that only 4 types of things can get here, array
    // subscript, array section, member expr, or DRE to a var decl (or the
    // former 3 wrapping a var-decl), so we should be able to assume this is
    // right.
    const auto *dre = cast<DeclRefExpr>(curVarExpr);
    const auto *vd = cast<VarDecl>(dre->getFoundDecl()->getCanonicalDecl());
    return {exprLoc, cgf.emitDeclRefLValue(dre).getPointer(), vd->getName(),
            std::move(bounds)};
  }

  template <typename BeforeOpTy, typename AfterOpTy>
  void addDataOperand(const Expr *varOperand, mlir::acc::DataClause dataClause,
                      bool structured, bool implicit) {
    DataOperandInfo opInfo = getDataOperandInfo(dirKind, varOperand);

    // TODO: OpenACC: we should comprehend the 'modifier-list' here for the data
    // operand. At the moment, we don't have a uniform way to assign these
    // properly, and the dialect cannot represent anything other than 'readonly'
    // and 'zero' on copyin/copyout/create, so for now, we skip it.

    auto beforeOp =
        builder.create<BeforeOpTy>(opInfo.beginLoc, opInfo.varValue, structured,
                                   implicit, opInfo.name, opInfo.bounds);
    operation.getDataClauseOperandsMutable().append(beforeOp.getResult());

    AfterOpTy afterOp;
    {
      mlir::OpBuilder::InsertionGuard guardCase(builder);
      builder.setInsertionPointAfter(operation);
      afterOp = builder.create<AfterOpTy>(opInfo.beginLoc, beforeOp.getResult(),
                                          opInfo.varValue, structured, implicit,
                                          opInfo.name, opInfo.bounds);
    }

    // Set the 'rest' of the info for both operations.
    beforeOp.setDataClause(dataClause);
    afterOp.setDataClause(dataClause);

    // Make sure we record these, so 'async' values can be updated later.
    dataOperands.push_back(beforeOp.getOperation());
    dataOperands.push_back(afterOp.getOperation());
  }

  // Helper function that covers for the fact that we don't have this function
  // on all operation types.
  mlir::ArrayAttr getAsyncOnlyAttr() {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>)
      return operation.getAsyncOnlyAttr();
    else if constexpr (isCombinedType<OpTy>)
      return operation.computeOp.getAsyncOnlyAttr();

    // Note: 'wait' has async as well, but it cannot have data clauses, so we
    // don't have to handle them here.

    llvm_unreachable("getting asyncOnly when clause not valid on operation?");
  }

  // Helper function that covers for the fact that we don't have this function
  // on all operation types.
  mlir::ArrayAttr getAsyncOperandsDeviceTypeAttr() {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>)
      return operation.getAsyncOperandsDeviceTypeAttr();
    else if constexpr (isCombinedType<OpTy>)
      return operation.computeOp.getAsyncOperandsDeviceTypeAttr();

    // Note: 'wait' has async as well, but it cannot have data clauses, so we
    // don't have to handle them here.

    llvm_unreachable(
        "getting asyncOperandsDeviceType when clause not valid on operation?");
  }

  // Helper function that covers for the fact that we don't have this function
  // on all operation types.
  mlir::OperandRange getAsyncOperands() {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>)
      return operation.getAsyncOperands();
    else if constexpr (isCombinedType<OpTy>)
      return operation.computeOp.getAsyncOperands();

    // Note: 'wait' has async as well, but it cannot have data clauses, so we
    // don't have to handle them here.

    llvm_unreachable(
        "getting asyncOperandsDeviceType when clause not valid on operation?");
  }

  // The 'data' clauses all require that we add the 'async' values from the
  // operation to them. We've collected the data operands along the way, so use
  // that list to get the current 'async' values.
  void updateDataOperandAsyncValues() {
    if (!hasAsyncClause || dataOperands.empty())
      return;

    for (mlir::Operation *dataOp : dataOperands) {
      llvm::TypeSwitch<mlir::Operation *, void>(dataOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto op) {
            op.setAsyncOnlyAttr(getAsyncOnlyAttr());
            op.setAsyncOperandsDeviceTypeAttr(getAsyncOperandsDeviceTypeAttr());
            op.getAsyncOperandsMutable().assign(getAsyncOperands());
          })
          .Default([&](mlir::Operation *) {
            llvm_unreachable("Not a data operation?");
          });
    }
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

  // The entry point for the CIR emitter. All users should use this rather than
  // 'visitClauseList', as this also handles the things that have to happen
  // 'after' the clauses are all visited.
  void emitClauses(ArrayRef<const OpenACCClause *> clauses) {
    this->VisitClauseList(clauses);
    updateDataOperandAsyncValues();
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
                                     emitIntExpr(clause.getIntExpr()),
                                     lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitNumGangsClause");
    }
  }

  void VisitVectorLengthClause(const OpenACCVectorLengthClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp,
                               mlir::acc::KernelsOp>) {
      operation.addVectorLengthOperand(builder.getContext(),
                                       emitIntExpr(clause.getIntExpr()),
                                       lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitVectorLengthClause");
    }
  }

  void VisitAsyncClause(const OpenACCAsyncClause &clause) {
    hasAsyncClause = true;
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp, mlir::acc::DataOp>) {
      if (!clause.hasIntExpr())
        operation.addAsyncOnly(builder.getContext(), lastDeviceTypeValues);
      else {

        mlir::Value intExpr;
        {
          // Async int exprs can be referenced by the data operands, which means
          // that the int-exprs have to appear before them.  IF there is a data
          // operand already, set the insertion point to 'before' it.
          mlir::OpBuilder::InsertionGuard guardCase(builder);
          if (!dataOperands.empty())
            builder.setInsertionPoint(dataOperands.front());
          intExpr = emitIntExpr(clause.getIntExpr());
        }
        operation.addAsyncOperand(builder.getContext(), intExpr,
                                  lastDeviceTypeValues);
      }
    } else if constexpr (isOneOfTypes<OpTy, mlir::acc::WaitOp>) {
      // Wait doesn't have a device_type, so its handling here is slightly
      // different.
      if (!clause.hasIntExpr())
        operation.setAsync(true);
      else
        operation.getAsyncOperandMutable().append(
            emitIntExpr(clause.getIntExpr()));
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Combined constructs remain. Data, enter data, exit data,
      // update constructs remain.
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
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. update construct remains.
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
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // 'if' applies to most of the constructs, but hold off on lowering them
      // until we can write tests/know what we're doing with codegen to make
      // sure we get it right.
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Enter data, exit data, host_data, update constructs
      // remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitDeviceNumClause(const OpenACCDeviceNumClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::InitOp, mlir::acc::ShutdownOp,
                               mlir::acc::SetOp>) {
      operation.getDeviceNumMutable().append(emitIntExpr(clause.getIntExpr()));
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
        values.push_back(emitIntExpr(E));

      operation.addNumGangsOperands(builder.getContext(), values,
                                    lastDeviceTypeValues);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitNumGangsClause");
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
          values.push_back(emitIntExpr(clause.getDevNumExpr()));
        for (const Expr *E : clause.getQueueIdExprs())
          values.push_back(emitIntExpr(E));
        operation.addWaitOperands(builder.getContext(), clause.hasDevNumExpr(),
                                  values, lastDeviceTypeValues);
      }
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. Enter data, exit data, update constructs remain.
      return clauseNotImplemented(clause);
    }
  }

  void VisitDefaultAsyncClause(const OpenACCDefaultAsyncClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::SetOp>) {
      operation.getDefaultAsyncMutable().append(
          emitIntExpr(clause.getIntExpr()));
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
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitCollapseClause");
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
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitTileClause");
    }
  }

  void VisitWorkerClause(const OpenACCWorkerClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::LoopOp>) {
      if (clause.hasIntExpr())
        operation.addWorkerNumOperand(builder.getContext(),
                                      emitIntExpr(clause.getIntExpr()),
                                      lastDeviceTypeValues);
      else
        operation.addEmptyWorker(builder.getContext(), lastDeviceTypeValues);

    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
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
                                   emitIntExpr(clause.getIntExpr()),
                                   lastDeviceTypeValues);
      else
        operation.addEmptyVector(builder.getContext(), lastDeviceTypeValues);

    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
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
            values.push_back(emitIntExpr(expr));
          }
        }

        operation.addGangOperands(builder.getContext(), lastDeviceTypeValues,
                                  argTypes, values);
      }
    } else if constexpr (isCombinedType<OpTy>) {
      applyToLoopOp(clause);
    } else {
      llvm_unreachable("Unknown construct kind in VisitGangClause");
    }
  }

  void VisitCopyClause(const OpenACCCopyClause &clause) {
    if constexpr (isOneOfTypes<OpTy, mlir::acc::ParallelOp, mlir::acc::SerialOp,
                               mlir::acc::KernelsOp>) {
      for (auto var : clause.getVarList())
        addDataOperand<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
            var, mlir::acc::DataClause::acc_copy, /*structured=*/true,
            /*implicit=*/false);
    } else if constexpr (isCombinedType<OpTy>) {
      applyToComputeOp(clause);
    } else {
      // TODO: When we've implemented this for everything, switch this to an
      // unreachable. data, declare, combined constructs remain.
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
} // namespace

template <typename Op>
void CIRGenFunction::emitOpenACCClauses(
    Op &op, OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
    ArrayRef<const OpenACCClause *> clauses) {
  mlir::OpBuilder::InsertionGuard guardCase(builder);

  // Sets insertion point before the 'op', since every new expression needs to
  // be before the operation.
  builder.setInsertionPoint(op);
  makeClauseEmitter(op, *this, builder, dirKind, dirLoc).emitClauses(clauses);
}

#define EXPL_SPEC(N)                                                           \
  template void CIRGenFunction::emitOpenACCClauses<N>(                         \
      N &, OpenACCDirectiveKind, SourceLocation,                               \
      ArrayRef<const OpenACCClause *>);
EXPL_SPEC(mlir::acc::ParallelOp)
EXPL_SPEC(mlir::acc::SerialOp)
EXPL_SPEC(mlir::acc::KernelsOp)
EXPL_SPEC(mlir::acc::LoopOp)
EXPL_SPEC(mlir::acc::DataOp)
EXPL_SPEC(mlir::acc::InitOp)
EXPL_SPEC(mlir::acc::ShutdownOp)
EXPL_SPEC(mlir::acc::SetOp)
EXPL_SPEC(mlir::acc::WaitOp)
#undef EXPL_SPEC

template <typename ComputeOp, typename LoopOp>
void CIRGenFunction::emitOpenACCClauses(
    ComputeOp &op, LoopOp &loopOp, OpenACCDirectiveKind dirKind,
    SourceLocation dirLoc, ArrayRef<const OpenACCClause *> clauses) {
  static_assert(std::is_same_v<mlir::acc::LoopOp, LoopOp>);

  CombinedConstructClauseInfo<ComputeOp> inf{op, loopOp};
  // We cannot set the insertion point here and do so in the emitter, but make
  // sure we reset it with the 'guard' anyway.
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  makeClauseEmitter(inf, *this, builder, dirKind, dirLoc).emitClauses(clauses);
}

#define EXPL_SPEC(N)                                                           \
  template void CIRGenFunction::emitOpenACCClauses<N, mlir::acc::LoopOp>(      \
      N &, mlir::acc::LoopOp &, OpenACCDirectiveKind, SourceLocation,          \
      ArrayRef<const OpenACCClause *>);

EXPL_SPEC(mlir::acc::ParallelOp)
EXPL_SPEC(mlir::acc::SerialOp)
EXPL_SPEC(mlir::acc::KernelsOp)
#undef EXPL_SPEC
