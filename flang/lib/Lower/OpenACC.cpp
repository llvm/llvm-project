//===-- OpenACC.cpp -- OpenACC directive lowering -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenACC.h"

#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/DirectivesCommon.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "flang-lower-openacc"

static llvm::cl::opt<bool> unwrapFirBox(
    "openacc-unwrap-fir-box",
    llvm::cl::desc(
        "Whether to use the address from fix.box in data clause operations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> generateDefaultBounds(
    "openacc-generate-default-bounds",
    llvm::cl::desc("Whether to generate default bounds for arrays."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> strideIncludeLowerExtent(
    "openacc-stride-include-lower-extent",
    llvm::cl::desc(
        "Whether to include the lower dimensions extents in the stride."),
    llvm::cl::init(true));

// Special value for * passed in device_type or gang clauses.
static constexpr std::int64_t starCst = -1;

static unsigned routineCounter = 0;
static constexpr llvm::StringRef accRoutinePrefix = "acc_routine_";
static constexpr llvm::StringRef accPrivateInitName = "acc.private.init";
static constexpr llvm::StringRef accReductionInitName = "acc.reduction.init";
static constexpr llvm::StringRef accFirDescriptorPostfix = "_desc";

static mlir::Location
genOperandLocation(Fortran::lower::AbstractConverter &converter,
                   const Fortran::parser::AccObject &accObject) {
  mlir::Location loc = converter.genUnknownLocation();
  Fortran::common::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            loc = converter.genLocation(designator.source);
          },
          [&](const Fortran::parser::Name &name) {
            loc = converter.genLocation(name.source);
          }},
      accObject.u);
  return loc;
}

static void addOperands(llvm::SmallVectorImpl<mlir::Value> &operands,
                        llvm::SmallVectorImpl<int32_t> &operandSegments,
                        llvm::ArrayRef<mlir::Value> clauseOperands) {
  operands.append(clauseOperands.begin(), clauseOperands.end());
  operandSegments.push_back(clauseOperands.size());
}

static void addOperand(llvm::SmallVectorImpl<mlir::Value> &operands,
                       llvm::SmallVectorImpl<int32_t> &operandSegments,
                       const mlir::Value &clauseOperand) {
  if (clauseOperand) {
    operands.push_back(clauseOperand);
    operandSegments.push_back(1);
  } else {
    operandSegments.push_back(0);
  }
}

template <typename Op>
static Op
createDataEntryOp(fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::Value baseAddr, std::stringstream &name,
                  mlir::SmallVector<mlir::Value> bounds, bool structured,
                  bool implicit, mlir::acc::DataClause dataClause,
                  mlir::Type retTy, llvm::ArrayRef<mlir::Value> async,
                  llvm::ArrayRef<mlir::Attribute> asyncDeviceTypes,
                  llvm::ArrayRef<mlir::Attribute> asyncOnlyDeviceTypes,
                  bool unwrapBoxAddr = false, mlir::Value isPresent = {}) {
  mlir::Value varPtrPtr;
  // The data clause may apply to either the box reference itself or the
  // pointer to the data it holds. So use `unwrapBoxAddr` to decide.
  // When we have a box value - assume it refers to the data inside box.
  if (unwrapFirBox &&
      ((fir::isBoxAddress(baseAddr.getType()) && unwrapBoxAddr) ||
       fir::isa_box_type(baseAddr.getType()))) {
    if (isPresent) {
      mlir::Type ifRetTy =
          mlir::cast<fir::BaseBoxType>(fir::unwrapRefType(baseAddr.getType()))
              .getEleTy();
      if (!fir::isa_ref_type(ifRetTy))
        ifRetTy = fir::ReferenceType::get(ifRetTy);
      baseAddr =
          builder
              .genIfOp(loc, {ifRetTy}, isPresent,
                       /*withElseRegion=*/true)
              .genThen([&]() {
                if (fir::isBoxAddress(baseAddr.getType()))
                  baseAddr = builder.create<fir::LoadOp>(loc, baseAddr);
                mlir::Value boxAddr =
                    builder.create<fir::BoxAddrOp>(loc, baseAddr);
                builder.create<fir::ResultOp>(loc, mlir::ValueRange{boxAddr});
              })
              .genElse([&] {
                mlir::Value absent =
                    builder.create<fir::AbsentOp>(loc, ifRetTy);
                builder.create<fir::ResultOp>(loc, mlir::ValueRange{absent});
              })
              .getResults()[0];
    } else {
      if (fir::isBoxAddress(baseAddr.getType()))
        baseAddr = builder.create<fir::LoadOp>(loc, baseAddr);
      baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    }
    retTy = baseAddr.getType();
  }

  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;

  addOperand(operands, operandSegments, baseAddr);
  addOperand(operands, operandSegments, varPtrPtr);
  addOperands(operands, operandSegments, bounds);
  addOperands(operands, operandSegments, async);

  Op op = builder.create<Op>(loc, retTy, operands);
  op.setNameAttr(builder.getStringAttr(name.str()));
  op.setStructured(structured);
  op.setImplicit(implicit);
  op.setDataClause(dataClause);
  if (auto mappableTy =
          mlir::dyn_cast<mlir::acc::MappableType>(baseAddr.getType())) {
    op.setVarType(baseAddr.getType());
  } else {
    assert(mlir::isa<mlir::acc::PointerLikeType>(baseAddr.getType()) &&
           "expected pointer-like");
    op.setVarType(mlir::cast<mlir::acc::PointerLikeType>(baseAddr.getType())
                      .getElementType());
  }

  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getDenseI32ArrayAttr(operandSegments));
  if (!asyncDeviceTypes.empty())
    op.setAsyncOperandsDeviceTypeAttr(builder.getArrayAttr(asyncDeviceTypes));
  if (!asyncOnlyDeviceTypes.empty())
    op.setAsyncOnlyAttr(builder.getArrayAttr(asyncOnlyDeviceTypes));
  return op;
}

static void addDeclareAttr(fir::FirOpBuilder &builder, mlir::Operation *op,
                           mlir::acc::DataClause clause) {
  if (!op)
    return;
  op->setAttr(mlir::acc::getDeclareAttrName(),
              mlir::acc::DeclareAttr::get(builder.getContext(),
                                          mlir::acc::DataClauseAttr::get(
                                              builder.getContext(), clause)));
}

static mlir::func::FuncOp
createDeclareFunc(mlir::OpBuilder &modBuilder, fir::FirOpBuilder &builder,
                  mlir::Location loc, llvm::StringRef funcName,
                  llvm::SmallVector<mlir::Type> argsTy = {},
                  llvm::SmallVector<mlir::Location> locs = {}) {
  auto funcTy = mlir::FunctionType::get(modBuilder.getContext(), argsTy, {});
  auto funcOp = modBuilder.create<mlir::func::FuncOp>(loc, funcName, funcTy);
  funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  builder.createBlock(&funcOp.getRegion(), funcOp.getRegion().end(), argsTy,
                      locs);
  builder.setInsertionPointToEnd(&funcOp.getRegion().back());
  builder.create<mlir::func::ReturnOp>(loc);
  builder.setInsertionPointToStart(&funcOp.getRegion().back());
  return funcOp;
}

template <typename Op>
static Op
createSimpleOp(fir::FirOpBuilder &builder, mlir::Location loc,
               const llvm::SmallVectorImpl<mlir::Value> &operands,
               const llvm::SmallVectorImpl<int32_t> &operandSegments) {
  llvm::ArrayRef<mlir::Type> argTy;
  Op op = builder.create<Op>(loc, argTy, operands);
  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getDenseI32ArrayAttr(operandSegments));
  return op;
}

template <typename EntryOp>
static void createDeclareAllocFuncWithArg(mlir::OpBuilder &modBuilder,
                                          fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Type descTy,
                                          llvm::StringRef funcNamePrefix,
                                          std::stringstream &asFortran,
                                          mlir::acc::DataClause clause) {
  auto crtInsPt = builder.saveInsertionPoint();
  std::stringstream registerFuncName;
  registerFuncName << funcNamePrefix.str()
                   << Fortran::lower::declarePostAllocSuffix.str();

  if (!mlir::isa<fir::ReferenceType>(descTy))
    descTy = fir::ReferenceType::get(descTy);
  auto registerFuncOp = createDeclareFunc(
      modBuilder, builder, loc, registerFuncName.str(), {descTy}, {loc});

  llvm::SmallVector<mlir::Value> bounds;
  std::stringstream asFortranDesc;
  asFortranDesc << asFortran.str();
  if (unwrapFirBox)
    asFortranDesc << accFirDescriptorPostfix.str();

  // Updating descriptor must occur before the mapping of the data so that
  // attached data pointer is not overwritten.
  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, registerFuncOp.getArgument(0), asFortranDesc, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, descTy,
          /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);

  if (unwrapFirBox) {
    mlir::Value desc =
        builder.create<fir::LoadOp>(loc, registerFuncOp.getArgument(0));
    fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, desc);
    addDeclareAttr(builder, boxAddrOp.getOperation(), clause);
    EntryOp entryOp = createDataEntryOp<EntryOp>(
        builder, loc, boxAddrOp.getResult(), asFortran, bounds,
        /*structured=*/false, /*implicit=*/false, clause, boxAddrOp.getType(),
        /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
    builder.create<mlir::acc::DeclareEnterOp>(
        loc, mlir::acc::DeclareTokenType::get(entryOp.getContext()),
        mlir::ValueRange(entryOp.getAccVar()));
  }

  modBuilder.setInsertionPointAfter(registerFuncOp);
  builder.restoreInsertionPoint(crtInsPt);
}

template <typename ExitOp>
static void createDeclareDeallocFuncWithArg(
    mlir::OpBuilder &modBuilder, fir::FirOpBuilder &builder, mlir::Location loc,
    mlir::Type descTy, llvm::StringRef funcNamePrefix,
    std::stringstream &asFortran, mlir::acc::DataClause clause) {
  auto crtInsPt = builder.saveInsertionPoint();
  // Generate the pre dealloc function.
  std::stringstream preDeallocFuncName;
  preDeallocFuncName << funcNamePrefix.str()
                     << Fortran::lower::declarePreDeallocSuffix.str();
  if (!mlir::isa<fir::ReferenceType>(descTy))
    descTy = fir::ReferenceType::get(descTy);
  auto preDeallocOp = createDeclareFunc(
      modBuilder, builder, loc, preDeallocFuncName.str(), {descTy}, {loc});

  mlir::Value var = preDeallocOp.getArgument(0);
  if (unwrapFirBox) {
    mlir::Value loadOp =
        builder.create<fir::LoadOp>(loc, preDeallocOp.getArgument(0));
    fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, loadOp);
    addDeclareAttr(builder, boxAddrOp.getOperation(), clause);
    var = boxAddrOp.getResult();
  }

  llvm::SmallVector<mlir::Value> bounds;
  mlir::acc::GetDevicePtrOp entryOp =
      createDataEntryOp<mlir::acc::GetDevicePtrOp>(
          builder, loc, var, asFortran, bounds,
          /*structured=*/false, /*implicit=*/false, clause, var.getType(),
          /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
  builder.create<mlir::acc::DeclareExitOp>(
      loc, mlir::Value{}, mlir::ValueRange(entryOp.getAccVar()));

  if constexpr (std::is_same_v<ExitOp, mlir::acc::CopyoutOp> ||
                std::is_same_v<ExitOp, mlir::acc::UpdateHostOp>)
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccVar(),
                           entryOp.getVar(), entryOp.getVarType(),
                           entryOp.getBounds(), entryOp.getAsyncOperands(),
                           entryOp.getAsyncOperandsDeviceTypeAttr(),
                           entryOp.getAsyncOnlyAttr(), entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));
  else
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccVar(),
                           entryOp.getBounds(), entryOp.getAsyncOperands(),
                           entryOp.getAsyncOperandsDeviceTypeAttr(),
                           entryOp.getAsyncOnlyAttr(), entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));

  // Generate the post dealloc function.
  modBuilder.setInsertionPointAfter(preDeallocOp);
  std::stringstream postDeallocFuncName;
  postDeallocFuncName << funcNamePrefix.str()
                      << Fortran::lower::declarePostDeallocSuffix.str();
  auto postDeallocOp = createDeclareFunc(
      modBuilder, builder, loc, postDeallocFuncName.str(), {descTy}, {loc});

  var = postDeallocOp.getArgument(0);
  if (unwrapFirBox) {
    var = builder.create<fir::LoadOp>(loc, postDeallocOp.getArgument(0));
    asFortran << accFirDescriptorPostfix.str();
  }

  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, var, asFortran, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, var.getType(),
          /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);
  modBuilder.setInsertionPointAfter(postDeallocOp);
  builder.restoreInsertionPoint(crtInsPt);
}

Fortran::semantics::Symbol &
getSymbolFromAccObject(const Fortran::parser::AccObject &accObject) {
  if (const auto *designator =
          std::get_if<Fortran::parser::Designator>(&accObject.u)) {
    if (const auto *name =
            Fortran::semantics::getDesignatorNameIfDataRef(*designator))
      return *name->symbol;
    if (const auto *arrayElement =
            Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                *designator)) {
      const Fortran::parser::Name &name =
          Fortran::parser::GetLastName(arrayElement->base);
      return *name.symbol;
    }
    if (const auto *component =
            Fortran::parser::Unwrap<Fortran::parser::StructureComponent>(
                *designator)) {
      return *component->component.symbol;
    }
  } else if (const auto *name =
                 std::get_if<Fortran::parser::Name>(&accObject.u)) {
    return *name->symbol;
  }
  llvm::report_fatal_error("Could not find symbol");
}

/// Used to generate atomic.read operation which is created in existing
/// location set by builder.
static inline void
genAtomicCaptureStatement(Fortran::lower::AbstractConverter &converter,
                          mlir::Value fromAddress, mlir::Value toAddress,
                          mlir::Type elementType, mlir::Location loc) {
  // Generate `atomic.read` operation for atomic assigment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  firOpBuilder.create<mlir::acc::AtomicReadOp>(
      loc, fromAddress, toAddress, mlir::TypeAttr::get(elementType));
}

/// Used to generate atomic.write operation which is created in existing
/// location set by builder.
static inline void
genAtomicWriteStatement(Fortran::lower::AbstractConverter &converter,
                        mlir::Value lhsAddr, mlir::Value rhsExpr,
                        mlir::Location loc,
                        mlir::Value *evaluatedExprValue = nullptr) {
  // Generate `atomic.write` operation for atomic assignment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::Type varType = fir::unwrapRefType(lhsAddr.getType());
  // Create a conversion outside the capture block.
  auto insertionPoint = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointAfter(rhsExpr.getDefiningOp());
  rhsExpr = firOpBuilder.createConvert(loc, varType, rhsExpr);
  firOpBuilder.restoreInsertionPoint(insertionPoint);

  firOpBuilder.create<mlir::acc::AtomicWriteOp>(loc, lhsAddr, rhsExpr);
}

/// Used to generate atomic.update operation which is created in existing
/// location set by builder.
static inline void genAtomicUpdateStatement(
    Fortran::lower::AbstractConverter &converter, mlir::Value lhsAddr,
    mlir::Type varType, const Fortran::parser::Variable &assignmentStmtVariable,
    const Fortran::parser::Expr &assignmentStmtExpr, mlir::Location loc,
    mlir::Operation *atomicCaptureOp = nullptr,
    Fortran::lower::StatementContext *atomicCaptureStmtCtx = nullptr) {
  // Generate `atomic.update` operation for atomic assignment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  //  Create the omp.atomic.update or acc.atomic.update operation
  //
  //  func.func @_QPsb() {
  //    %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFsbEa"}
  //    %1 = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFsbEb"}
  //    %2 = fir.load %1 : !fir.ref<i32>
  //    omp.atomic.update   %0 : !fir.ref<i32> {
  //    ^bb0(%arg0: i32):
  //      %3 = arith.addi %arg0, %2 : i32
  //      omp.yield(%3 : i32)
  //    }
  //    return
  //  }

  auto getArgExpression =
      [](std::list<Fortran::parser::ActualArgSpec>::const_iterator it) {
        const auto &arg{std::get<Fortran::parser::ActualArg>((*it).t)};
        const auto *parserExpr{
            std::get_if<Fortran::common::Indirection<Fortran::parser::Expr>>(
                &arg.u)};
        return parserExpr;
      };

  // Lower any non atomic sub-expression before the atomic operation, and
  // map its lowered value to the semantic representation.
  Fortran::lower::ExprToValueMap exprValueOverrides;
  // Max and min intrinsics can have a list of Args. Hence we need a list
  // of nonAtomicSubExprs to hoist. Currently, only the load is hoisted.
  llvm::SmallVector<const Fortran::lower::SomeExpr *> nonAtomicSubExprs;
  Fortran::common::visit(
      Fortran::common::visitors{
          [&](const Fortran::common::Indirection<
              Fortran::parser::FunctionReference> &funcRef) -> void {
            const auto &args{
                std::get<std::list<Fortran::parser::ActualArgSpec>>(
                    funcRef.value().v.t)};
            std::list<Fortran::parser::ActualArgSpec>::const_iterator beginIt =
                args.begin();
            std::list<Fortran::parser::ActualArgSpec>::const_iterator endIt =
                args.end();
            const auto *exprFirst{getArgExpression(beginIt)};
            if (exprFirst && exprFirst->value().source ==
                                 assignmentStmtVariable.GetSource()) {
              // Add everything except the first
              beginIt++;
            } else {
              // Add everything except the last
              endIt--;
            }
            std::list<Fortran::parser::ActualArgSpec>::const_iterator it;
            for (it = beginIt; it != endIt; it++) {
              const Fortran::common::Indirection<Fortran::parser::Expr> *expr =
                  getArgExpression(it);
              if (expr)
                nonAtomicSubExprs.push_back(Fortran::semantics::GetExpr(*expr));
            }
          },
          [&](const auto &op) -> void {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_base_of<
                              Fortran::parser::Expr::IntrinsicBinary,
                              T>::value) {
              const auto &exprLeft{std::get<0>(op.t)};
              const auto &exprRight{std::get<1>(op.t)};
              if (exprLeft.value().source == assignmentStmtVariable.GetSource())
                nonAtomicSubExprs.push_back(
                    Fortran::semantics::GetExpr(exprRight));
              else
                nonAtomicSubExprs.push_back(
                    Fortran::semantics::GetExpr(exprLeft));
            }
          },
      },
      assignmentStmtExpr.u);
  Fortran::lower::StatementContext nonAtomicStmtCtx;
  Fortran::lower::StatementContext *stmtCtxPtr = &nonAtomicStmtCtx;
  if (!nonAtomicSubExprs.empty()) {
    // Generate non atomic part before all the atomic operations.
    auto insertionPoint = firOpBuilder.saveInsertionPoint();
    if (atomicCaptureOp) {
      assert(atomicCaptureStmtCtx && "must specify statement context");
      firOpBuilder.setInsertionPoint(atomicCaptureOp);
      // Any clean-ups associated with the expression lowering
      // must also be generated outside of the atomic update operation
      // and after the atomic capture operation.
      // The atomicCaptureStmtCtx will be finalized at the end
      // of the atomic capture operation generation.
      stmtCtxPtr = atomicCaptureStmtCtx;
    }
    mlir::Value nonAtomicVal;
    for (auto *nonAtomicSubExpr : nonAtomicSubExprs) {
      nonAtomicVal = fir::getBase(converter.genExprValue(
          currentLocation, *nonAtomicSubExpr, *stmtCtxPtr));
      exprValueOverrides.try_emplace(nonAtomicSubExpr, nonAtomicVal);
    }
    if (atomicCaptureOp)
      firOpBuilder.restoreInsertionPoint(insertionPoint);
  }

  mlir::Operation *atomicUpdateOp = nullptr;
  atomicUpdateOp =
      firOpBuilder.create<mlir::acc::AtomicUpdateOp>(currentLocation, lhsAddr);

  llvm::SmallVector<mlir::Type> varTys = {varType};
  llvm::SmallVector<mlir::Location> locs = {currentLocation};
  firOpBuilder.createBlock(&atomicUpdateOp->getRegion(0), {}, varTys, locs);
  mlir::Value val =
      fir::getBase(atomicUpdateOp->getRegion(0).front().getArgument(0));

  exprValueOverrides.try_emplace(
      Fortran::semantics::GetExpr(assignmentStmtVariable), val);
  {
    // statement context inside the atomic block.
    converter.overrideExprValues(&exprValueOverrides);
    Fortran::lower::StatementContext atomicStmtCtx;
    mlir::Value rhsExpr = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(assignmentStmtExpr), atomicStmtCtx));
    mlir::Value convertResult =
        firOpBuilder.createConvert(currentLocation, varType, rhsExpr);
    firOpBuilder.create<mlir::acc::YieldOp>(currentLocation, convertResult);
    converter.resetExprOverrides();
  }
  firOpBuilder.setInsertionPointAfter(atomicUpdateOp);
}

/// Processes an atomic construct with write clause.
void genAtomicWrite(Fortran::lower::AbstractConverter &converter,
                    const Fortran::parser::AccAtomicWrite &atomicWrite,
                    mlir::Location loc) {
  const Fortran::parser::AssignmentStmt &stmt =
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicWrite.t)
          .statement;
  const Fortran::evaluate::Assignment &assign = *stmt.typedAssignment->v;
  Fortran::lower::StatementContext stmtCtx;
  // Get the value and address of atomic write operands.
  mlir::Value rhsExpr =
      fir::getBase(converter.genExprValue(assign.rhs, stmtCtx));
  mlir::Value lhsAddr =
      fir::getBase(converter.genExprAddr(assign.lhs, stmtCtx));
  genAtomicWriteStatement(converter, lhsAddr, rhsExpr, loc);
}

/// Processes an atomic construct with read clause.
void genAtomicRead(Fortran::lower::AbstractConverter &converter,
                   const Fortran::parser::AccAtomicRead &atomicRead,
                   mlir::Location loc) {
  const auto &assignmentStmtExpr = std::get<Fortran::parser::Expr>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicRead.t)
          .statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicRead.t)
          .statement.t);

  Fortran::lower::StatementContext stmtCtx;
  const Fortran::semantics::SomeExpr &fromExpr =
      *Fortran::semantics::GetExpr(assignmentStmtExpr);
  mlir::Type elementType = converter.genType(fromExpr);
  mlir::Value fromAddress =
      fir::getBase(converter.genExprAddr(fromExpr, stmtCtx));
  mlir::Value toAddress = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  genAtomicCaptureStatement(converter, fromAddress, toAddress, elementType,
                            loc);
}

/// Processes an atomic construct with update clause.
void genAtomicUpdate(Fortran::lower::AbstractConverter &converter,
                     const Fortran::parser::AccAtomicUpdate &atomicUpdate,
                     mlir::Location loc) {
  const auto &assignmentStmtExpr = std::get<Fortran::parser::Expr>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicUpdate.t)
          .statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicUpdate.t)
          .statement.t);

  Fortran::lower::StatementContext stmtCtx;
  mlir::Value lhsAddr = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  mlir::Type varType = fir::unwrapRefType(lhsAddr.getType());
  genAtomicUpdateStatement(converter, lhsAddr, varType, assignmentStmtVariable,
                           assignmentStmtExpr, loc);
}

/// Processes an atomic construct with capture clause.
void genAtomicCapture(Fortran::lower::AbstractConverter &converter,
                      const Fortran::parser::AccAtomicCapture &atomicCapture,
                      mlir::Location loc) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  const Fortran::parser::AssignmentStmt &stmt1 =
      std::get<Fortran::parser::AccAtomicCapture::Stmt1>(atomicCapture.t)
          .v.statement;
  const Fortran::evaluate::Assignment &assign1 = *stmt1.typedAssignment->v;
  const auto &stmt1Var{std::get<Fortran::parser::Variable>(stmt1.t)};
  const auto &stmt1Expr{std::get<Fortran::parser::Expr>(stmt1.t)};
  const Fortran::parser::AssignmentStmt &stmt2 =
      std::get<Fortran::parser::AccAtomicCapture::Stmt2>(atomicCapture.t)
          .v.statement;
  const Fortran::evaluate::Assignment &assign2 = *stmt2.typedAssignment->v;
  const auto &stmt2Var{std::get<Fortran::parser::Variable>(stmt2.t)};
  const auto &stmt2Expr{std::get<Fortran::parser::Expr>(stmt2.t)};

  // Pre-evaluate expressions to be used in the various operations inside
  // `atomic.capture` since it is not desirable to have anything other than
  // a `atomic.read`, `atomic.write`, or `atomic.update` operation
  // inside `atomic.capture`
  Fortran::lower::StatementContext stmtCtx;
  // LHS evaluations are common to all combinations of `atomic.capture`
  mlir::Value stmt1LHSArg =
      fir::getBase(converter.genExprAddr(assign1.lhs, stmtCtx));
  mlir::Value stmt2LHSArg =
      fir::getBase(converter.genExprAddr(assign2.lhs, stmtCtx));

  // Type information used in generation of `atomic.update` operation
  mlir::Type stmt1VarType =
      fir::getBase(converter.genExprValue(assign1.lhs, stmtCtx)).getType();
  mlir::Type stmt2VarType =
      fir::getBase(converter.genExprValue(assign2.lhs, stmtCtx)).getType();

  mlir::Operation *atomicCaptureOp = nullptr;
  atomicCaptureOp = firOpBuilder.create<mlir::acc::AtomicCaptureOp>(loc);

  firOpBuilder.createBlock(&(atomicCaptureOp->getRegion(0)));
  mlir::Block &block = atomicCaptureOp->getRegion(0).back();
  firOpBuilder.setInsertionPointToStart(&block);
  if (Fortran::parser::CheckForSingleVariableOnRHS(stmt1)) {
    if (Fortran::evaluate::CheckForSymbolMatch(
            Fortran::semantics::GetExpr(stmt2Var),
            Fortran::semantics::GetExpr(stmt2Expr))) {
      // Atomic capture construct is of the form [capture-stmt, update-stmt]
      const Fortran::semantics::SomeExpr &fromExpr =
          *Fortran::semantics::GetExpr(stmt1Expr);
      mlir::Type elementType = converter.genType(fromExpr);
      genAtomicCaptureStatement(converter, stmt2LHSArg, stmt1LHSArg,
                                elementType, loc);
      genAtomicUpdateStatement(converter, stmt2LHSArg, stmt2VarType, stmt2Var,
                               stmt2Expr, loc, atomicCaptureOp, &stmtCtx);
    } else {
      // Atomic capture construct is of the form [capture-stmt, write-stmt]
      firOpBuilder.setInsertionPoint(atomicCaptureOp);
      mlir::Value stmt2RHSArg =
          fir::getBase(converter.genExprValue(assign2.rhs, stmtCtx));
      firOpBuilder.setInsertionPointToStart(&block);
      const Fortran::semantics::SomeExpr &fromExpr =
          *Fortran::semantics::GetExpr(stmt1Expr);
      mlir::Type elementType = converter.genType(fromExpr);
      genAtomicCaptureStatement(converter, stmt2LHSArg, stmt1LHSArg,
                                elementType, loc);
      genAtomicWriteStatement(converter, stmt2LHSArg, stmt2RHSArg, loc);
    }
  } else {
    // Atomic capture construct is of the form [update-stmt, capture-stmt]
    const Fortran::semantics::SomeExpr &fromExpr =
        *Fortran::semantics::GetExpr(stmt2Expr);
    mlir::Type elementType = converter.genType(fromExpr);
    genAtomicUpdateStatement(converter, stmt1LHSArg, stmt1VarType, stmt1Var,
                             stmt1Expr, loc, atomicCaptureOp, &stmtCtx);
    genAtomicCaptureStatement(converter, stmt1LHSArg, stmt2LHSArg, elementType,
                              loc);
  }
  firOpBuilder.setInsertionPointToEnd(&block);
  firOpBuilder.create<mlir::acc::TerminatorOp>(loc);
  // The clean-ups associated with the statements inside the capture
  // construct must be generated after the AtomicCaptureOp.
  firOpBuilder.setInsertionPointAfter(atomicCaptureOp);
}

template <typename Op>
static void
genDataOperandOperations(const Fortran::parser::AccObjectList &objectList,
                         Fortran::lower::AbstractConverter &converter,
                         Fortran::semantics::SemanticsContext &semanticsContext,
                         Fortran::lower::StatementContext &stmtCtx,
                         llvm::SmallVectorImpl<mlir::Value> &dataOperands,
                         mlir::acc::DataClause dataClause, bool structured,
                         bool implicit, llvm::ArrayRef<mlir::Value> async,
                         llvm::ArrayRef<mlir::Attribute> asyncDeviceTypes,
                         llvm::ArrayRef<mlir::Attribute> asyncOnlyDeviceTypes,
                         bool setDeclareAttr = false) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::evaluate::ExpressionAnalyzer ea{semanticsContext};
  for (const auto &accObject : objectList.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    Fortran::semantics::Symbol &symbol = getSymbolFromAccObject(accObject);
    Fortran::semantics::MaybeExpr designator = Fortran::common::visit(
        [&](auto &&s) { return ea.Analyze(s); }, accObject.u);
    fir::factory::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            mlir::acc::DataBoundsOp, mlir::acc::DataBoundsType>(
            converter, builder, semanticsContext, stmtCtx, symbol, designator,
            operandLocation, asFortran, bounds,
            /*treatIndexAsSection=*/true, /*unwrapFirBox=*/unwrapFirBox,
            /*genDefaultBounds=*/generateDefaultBounds,
            /*strideIncludeLowerExtent=*/strideIncludeLowerExtent);
    LLVM_DEBUG(llvm::dbgs() << __func__ << "\n"; info.dump(llvm::dbgs()));

    // If the input value is optional and is not a descriptor, we use the
    // rawInput directly.
    mlir::Value baseAddr = ((fir::unwrapRefType(info.addr.getType()) !=
                             fir::unwrapRefType(info.rawInput.getType())) &&
                            info.isPresent)
                               ? info.rawInput
                               : info.addr;
    Op op = createDataEntryOp<Op>(
        builder, operandLocation, baseAddr, asFortran, bounds, structured,
        implicit, dataClause, baseAddr.getType(), async, asyncDeviceTypes,
        asyncOnlyDeviceTypes, /*unwrapBoxAddr=*/true, info.isPresent);
    dataOperands.push_back(op.getAccVar());
  }
}

template <typename EntryOp, typename ExitOp>
static void genDeclareDataOperandOperations(
    const Fortran::parser::AccObjectList &objectList,
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    llvm::SmallVectorImpl<mlir::Value> &dataOperands,
    mlir::acc::DataClause dataClause, bool structured, bool implicit) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::evaluate::ExpressionAnalyzer ea{semanticsContext};
  for (const auto &accObject : objectList.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    Fortran::semantics::Symbol &symbol = getSymbolFromAccObject(accObject);
    Fortran::semantics::MaybeExpr designator = Fortran::common::visit(
        [&](auto &&s) { return ea.Analyze(s); }, accObject.u);
    fir::factory::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            mlir::acc::DataBoundsOp, mlir::acc::DataBoundsType>(
            converter, builder, semanticsContext, stmtCtx, symbol, designator,
            operandLocation, asFortran, bounds,
            /*treatIndexAsSection=*/true, /*unwrapFirBox=*/unwrapFirBox,
            /*genDefaultBounds=*/generateDefaultBounds,
            /*strideIncludeLowerExtent=*/strideIncludeLowerExtent);
    LLVM_DEBUG(llvm::dbgs() << __func__ << "\n"; info.dump(llvm::dbgs()));
    EntryOp op = createDataEntryOp<EntryOp>(
        builder, operandLocation, info.addr, asFortran, bounds, structured,
        implicit, dataClause, info.addr.getType(),
        /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
    dataOperands.push_back(op.getAccVar());
    addDeclareAttr(builder, op.getVar().getDefiningOp(), dataClause);
    if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(info.addr.getType()))) {
      mlir::OpBuilder modBuilder(builder.getModule().getBodyRegion());
      modBuilder.setInsertionPointAfter(builder.getFunction());
      std::string prefix = converter.mangleName(symbol);
      createDeclareAllocFuncWithArg<EntryOp>(
          modBuilder, builder, operandLocation, info.addr.getType(), prefix,
          asFortran, dataClause);
      if constexpr (!std::is_same_v<EntryOp, ExitOp>)
        createDeclareDeallocFuncWithArg<ExitOp>(
            modBuilder, builder, operandLocation, info.addr.getType(), prefix,
            asFortran, dataClause);
    }
  }
}

template <typename EntryOp, typename ExitOp, typename Clause>
static void genDeclareDataOperandOperationsWithModifier(
    const Clause *x, Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::parser::AccDataModifier::Modifier mod,
    llvm::SmallVectorImpl<mlir::Value> &dataClauseOperands,
    const mlir::acc::DataClause clause,
    const mlir::acc::DataClause clauseWithModifier) {
  const Fortran::parser::AccObjectListWithModifier &listWithModifier = x->v;
  const auto &accObjectList =
      std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
  const auto &modifier =
      std::get<std::optional<Fortran::parser::AccDataModifier>>(
          listWithModifier.t);
  mlir::acc::DataClause dataClause =
      (modifier && (*modifier).v == mod) ? clauseWithModifier : clause;
  genDeclareDataOperandOperations<EntryOp, ExitOp>(
      accObjectList, converter, semanticsContext, stmtCtx, dataClauseOperands,
      dataClause,
      /*structured=*/true, /*implicit=*/false);
}

template <typename EntryOp, typename ExitOp>
static void
genDataExitOperations(fir::FirOpBuilder &builder,
                      llvm::SmallVector<mlir::Value> operands, bool structured,
                      std::optional<mlir::Location> exitLoc = std::nullopt) {
  for (mlir::Value operand : operands) {
    auto entryOp = mlir::dyn_cast_or_null<EntryOp>(operand.getDefiningOp());
    assert(entryOp && "data entry op expected");
    mlir::Location opLoc = exitLoc ? *exitLoc : entryOp.getLoc();
    if constexpr (std::is_same_v<ExitOp, mlir::acc::CopyoutOp> ||
                  std::is_same_v<ExitOp, mlir::acc::UpdateHostOp>)
      builder.create<ExitOp>(
          opLoc, entryOp.getAccVar(), entryOp.getVar(), entryOp.getVarType(),
          entryOp.getBounds(), entryOp.getAsyncOperands(),
          entryOp.getAsyncOperandsDeviceTypeAttr(), entryOp.getAsyncOnlyAttr(),
          entryOp.getDataClause(), structured, entryOp.getImplicit(),
          builder.getStringAttr(*entryOp.getName()));
    else
      builder.create<ExitOp>(
          opLoc, entryOp.getAccVar(), entryOp.getBounds(),
          entryOp.getAsyncOperands(), entryOp.getAsyncOperandsDeviceTypeAttr(),
          entryOp.getAsyncOnlyAttr(), entryOp.getDataClause(), structured,
          entryOp.getImplicit(), builder.getStringAttr(*entryOp.getName()));
  }
}

fir::ShapeOp genShapeOp(mlir::OpBuilder &builder, fir::SequenceType seqTy,
                        mlir::Location loc) {
  llvm::SmallVector<mlir::Value> extents;
  mlir::Type idxTy = builder.getIndexType();
  for (auto extent : seqTy.getShape())
    extents.push_back(builder.create<mlir::arith::ConstantOp>(
        loc, idxTy, builder.getIntegerAttr(idxTy, extent)));
  return builder.create<fir::ShapeOp>(loc, extents);
}

/// Get the initial value for reduction operator.
template <typename R>
static R getReductionInitValue(mlir::acc::ReductionOperator op, mlir::Type ty) {
  if (op == mlir::acc::ReductionOperator::AccMin) {
    // min init value -> largest
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt::getSignedMaxValue(ty.getIntOrFloatBitWidth());
    }
    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty);
      assert(floatTy && "expect float type");
      return llvm::APFloat::getLargest(floatTy.getFloatSemantics(),
                                       /*negative=*/false);
    }
  } else if (op == mlir::acc::ReductionOperator::AccMax) {
    // max init value -> smallest
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt::getSignedMinValue(ty.getIntOrFloatBitWidth());
    }
    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty);
      assert(floatTy && "expect float type");
      return llvm::APFloat::getSmallest(floatTy.getFloatSemantics(),
                                        /*negative=*/true);
    }
  } else if (op == mlir::acc::ReductionOperator::AccIand) {
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer type");
      unsigned bits = ty.getIntOrFloatBitWidth();
      return llvm::APInt::getAllOnes(bits);
    }
  } else {
    assert(op != mlir::acc::ReductionOperator::AccNone);
    // +, ior, ieor init value -> 0
    // * init value -> 1
    int64_t value = (op == mlir::acc::ReductionOperator::AccMul) ? 1 : 0;
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt(ty.getIntOrFloatBitWidth(), value, true);
    }

    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      assert(mlir::isa<mlir::FloatType>(ty) && "expect float type");
      auto floatTy = mlir::dyn_cast<mlir::FloatType>(ty);
      return llvm::APFloat(floatTy.getFloatSemantics(), value);
    }

    if constexpr (std::is_same_v<R, int64_t>)
      return value;
  }
  llvm_unreachable("OpenACC reduction unsupported type");
}

/// Return a constant with the initial value for the reduction operator and
/// type combination.
static mlir::Value getReductionInitValue(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Type ty,
                                         mlir::acc::ReductionOperator op) {
  if (op == mlir::acc::ReductionOperator::AccLand ||
      op == mlir::acc::ReductionOperator::AccLor ||
      op == mlir::acc::ReductionOperator::AccEqv ||
      op == mlir::acc::ReductionOperator::AccNeqv) {
    assert(mlir::isa<fir::LogicalType>(ty) && "expect fir.logical type");
    bool value = true; // .true. for .and. and .eqv.
    if (op == mlir::acc::ReductionOperator::AccLor ||
        op == mlir::acc::ReductionOperator::AccNeqv)
      value = false; // .false. for .or. and .neqv.
    return builder.createBool(loc, value);
  }
  if (ty.isIntOrIndex())
    return builder.create<mlir::arith::ConstantOp>(
        loc, ty,
        builder.getIntegerAttr(ty, getReductionInitValue<llvm::APInt>(op, ty)));
  if (op == mlir::acc::ReductionOperator::AccMin ||
      op == mlir::acc::ReductionOperator::AccMax) {
    if (mlir::isa<mlir::ComplexType>(ty))
      llvm::report_fatal_error(
          "min/max reduction not supported for complex type");
    if (auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty))
      return builder.create<mlir::arith::ConstantOp>(
          loc, ty,
          builder.getFloatAttr(ty,
                               getReductionInitValue<llvm::APFloat>(op, ty)));
  } else if (auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty)) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, ty,
        builder.getFloatAttr(ty, getReductionInitValue<int64_t>(op, ty)));
  } else if (auto cmplxTy = mlir::dyn_cast_or_null<mlir::ComplexType>(ty)) {
    mlir::Type floatTy = cmplxTy.getElementType();
    mlir::Value realInit = builder.createRealConstant(
        loc, floatTy, getReductionInitValue<int64_t>(op, cmplxTy));
    mlir::Value imagInit = builder.createRealConstant(loc, floatTy, 0.0);
    return fir::factory::Complex{builder, loc}.createComplex(cmplxTy, realInit,
                                                             imagInit);
  }

  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
    return getReductionInitValue(builder, loc, seqTy.getEleTy(), op);

  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty))
    return getReductionInitValue(builder, loc, boxTy.getEleTy(), op);

  if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
    return getReductionInitValue(builder, loc, heapTy.getEleTy(), op);

  if (auto ptrTy = mlir::dyn_cast<fir::PointerType>(ty))
    return getReductionInitValue(builder, loc, ptrTy.getEleTy(), op);

  llvm::report_fatal_error("Unsupported OpenACC reduction type");
}

template <typename RecipeOp>
static void genPrivateLikeInitRegion(fir::FirOpBuilder &builder,
                                     RecipeOp recipe, mlir::Type argTy,
                                     mlir::Location loc,
                                     mlir::Value initValue) {
  mlir::Value retVal = recipe.getInitRegion().front().getArgument(0);
  mlir::Type unwrappedTy = fir::unwrapRefType(argTy);

  llvm::StringRef initName;
  if constexpr (std::is_same_v<RecipeOp, mlir::acc::ReductionRecipeOp>)
    initName = accReductionInitName;
  else
    initName = accPrivateInitName;

  auto getDeclareOpForType = [&](mlir::Type ty) -> hlfir::DeclareOp {
    auto alloca = builder.create<fir::AllocaOp>(loc, ty);
    return builder.create<hlfir::DeclareOp>(
        loc, alloca, initName, /*shape=*/nullptr, llvm::ArrayRef<mlir::Value>{},
        /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});
  };

  if (fir::isa_trivial(unwrappedTy)) {
    auto declareOp = getDeclareOpForType(unwrappedTy);
    if (initValue) {
      auto convert = builder.createConvert(loc, unwrappedTy, initValue);
      builder.create<fir::StoreOp>(loc, convert, declareOp.getBase());
    }
    retVal = declareOp.getBase();
  } else if (auto seqTy =
                 mlir::dyn_cast_or_null<fir::SequenceType>(unwrappedTy)) {
    if (fir::isa_trivial(seqTy.getEleTy())) {
      mlir::Value shape;
      llvm::SmallVector<mlir::Value> extents;
      if (seqTy.hasDynamicExtents()) {
        // Extents are passed as block arguments. First argument is the
        // original value.
        for (unsigned i = 1; i < recipe.getInitRegion().getArguments().size();
             ++i)
          extents.push_back(recipe.getInitRegion().getArgument(i));
        shape = builder.create<fir::ShapeOp>(loc, extents);
      } else {
        shape = genShapeOp(builder, seqTy, loc);
      }
      auto alloca = builder.create<fir::AllocaOp>(
          loc, seqTy, /*typeparams=*/mlir::ValueRange{}, extents);
      auto declareOp = builder.create<hlfir::DeclareOp>(
          loc, alloca, initName, shape, llvm::ArrayRef<mlir::Value>{},
          /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});

      if (initValue) {
        mlir::Type idxTy = builder.getIndexType();
        mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
        llvm::SmallVector<fir::DoLoopOp> loops;
        llvm::SmallVector<mlir::Value> ivs;

        if (seqTy.hasDynamicExtents()) {
          builder.create<hlfir::AssignOp>(loc, initValue, declareOp.getBase());
        } else {
          for (auto ext : seqTy.getShape()) {
            auto lb = builder.createIntegerConstant(loc, idxTy, 0);
            auto ub = builder.createIntegerConstant(loc, idxTy, ext - 1);
            auto step = builder.createIntegerConstant(loc, idxTy, 1);
            auto loop = builder.create<fir::DoLoopOp>(loc, lb, ub, step,
                                                      /*unordered=*/false);
            builder.setInsertionPointToStart(loop.getBody());
            loops.push_back(loop);
            ivs.push_back(loop.getInductionVar());
          }
          auto coord = builder.create<fir::CoordinateOp>(
              loc, refTy, declareOp.getBase(), ivs);
          builder.create<fir::StoreOp>(loc, initValue, coord);
          builder.setInsertionPointAfter(loops[0]);
        }
      }
      retVal = declareOp.getBase();
    }
  } else if (auto boxTy =
                 mlir::dyn_cast_or_null<fir::BaseBoxType>(unwrappedTy)) {
    mlir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
    if (fir::isa_trivial(innerTy)) {
      retVal = getDeclareOpForType(unwrappedTy).getBase();
    } else if (mlir::isa<fir::SequenceType>(innerTy)) {
      fir::FirOpBuilder firBuilder{builder, recipe.getOperation()};
      hlfir::Entity source = hlfir::Entity{retVal};
      auto [temp, cleanup] = hlfir::createTempFromMold(loc, firBuilder, source);
      if (fir::isa_ref_type(argTy)) {
        // When the temp is created - it is not a reference - thus we can
        // end up with a type inconsistency. Therefore ensure storage is created
        // for it.
        retVal = getDeclareOpForType(unwrappedTy).getBase();
        mlir::Value storeDst = retVal;
        if (fir::unwrapRefType(retVal.getType()) != temp.getType()) {
          // `createTempFromMold` makes the unfortunate choice to lose the
          // `fir.heap` and `fir.ptr` types when wrapping with a box. Namely,
          // when wrapping a `fir.heap<fir.array>`, it will create instead a
          // `fir.box<fir.array>`. Cast here to deal with this inconsistency.
          storeDst = firBuilder.createConvert(
              loc, firBuilder.getRefType(temp.getType()), retVal);
        }
        builder.create<fir::StoreOp>(loc, temp, storeDst);
      } else {
        retVal = temp;
      }
    } else {
      TODO(loc, "Unsupported boxed type for OpenACC private-like recipe");
    }
    if (initValue) {
      builder.create<hlfir::AssignOp>(loc, initValue, retVal);
    }
  }
  builder.create<mlir::acc::YieldOp>(loc, retVal);
}

template <typename RecipeOp>
static RecipeOp genRecipeOp(
    fir::FirOpBuilder &builder, mlir::ModuleOp mod, llvm::StringRef recipeName,
    mlir::Location loc, mlir::Type ty,
    mlir::acc::ReductionOperator op = mlir::acc::ReductionOperator::AccNone) {
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  RecipeOp recipe;
  if constexpr (std::is_same_v<RecipeOp, mlir::acc::ReductionRecipeOp>) {
    recipe = modBuilder.create<mlir::acc::ReductionRecipeOp>(loc, recipeName,
                                                             ty, op);
  } else {
    recipe = modBuilder.create<RecipeOp>(loc, recipeName, ty);
  }

  llvm::SmallVector<mlir::Type> argsTy{ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc};
  if (auto refTy = mlir::dyn_cast_or_null<fir::ReferenceType>(ty)) {
    if (auto seqTy =
            mlir::dyn_cast_or_null<fir::SequenceType>(refTy.getEleTy())) {
      if (seqTy.hasDynamicExtents()) {
        mlir::Type idxTy = builder.getIndexType();
        for (unsigned i = 0; i < seqTy.getDimension(); ++i) {
          argsTy.push_back(idxTy);
          argsLoc.push_back(loc);
        }
      }
    }
  }
  builder.createBlock(&recipe.getInitRegion(), recipe.getInitRegion().end(),
                      argsTy, argsLoc);
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  mlir::Value initValue;
  if constexpr (std::is_same_v<RecipeOp, mlir::acc::ReductionRecipeOp>) {
    assert(op != mlir::acc::ReductionOperator::AccNone);
    initValue = getReductionInitValue(builder, loc, fir::unwrapRefType(ty), op);
  }
  genPrivateLikeInitRegion<RecipeOp>(builder, recipe, ty, loc, initValue);
  return recipe;
}

mlir::acc::PrivateRecipeOp
Fortran::lower::createOrGetPrivateRecipe(fir::FirOpBuilder &builder,
                                         llvm::StringRef recipeName,
                                         mlir::Location loc, mlir::Type ty) {
  mlir::ModuleOp mod =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  if (auto recipe = mod.lookupSymbol<mlir::acc::PrivateRecipeOp>(recipeName))
    return recipe;

  auto ip = builder.saveInsertionPoint();
  auto recipe = genRecipeOp<mlir::acc::PrivateRecipeOp>(builder, mod,
                                                        recipeName, loc, ty);
  builder.restoreInsertionPoint(ip);
  return recipe;
}

/// Check if the DataBoundsOp is a constant bound (lb and ub are constants or
/// extent is a constant).
bool isConstantBound(mlir::acc::DataBoundsOp &op) {
  if (op.getLowerbound() && fir::getIntIfConstant(op.getLowerbound()) &&
      op.getUpperbound() && fir::getIntIfConstant(op.getUpperbound()))
    return true;
  if (op.getExtent() && fir::getIntIfConstant(op.getExtent()))
    return true;
  return false;
}

/// Return true iff all the bounds are expressed with constant values.
bool areAllBoundConstant(const llvm::SmallVector<mlir::Value> &bounds) {
  for (auto bound : bounds) {
    auto dataBound =
        mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
    assert(dataBound && "Must be DataBoundOp operation");
    if (!isConstantBound(dataBound))
      return false;
  }
  return true;
}

static llvm::SmallVector<mlir::Value>
genConstantBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::acc::DataBoundsOp &dataBound) {
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value lb, ub, step;
  if (dataBound.getLowerbound() &&
      fir::getIntIfConstant(dataBound.getLowerbound()) &&
      dataBound.getUpperbound() &&
      fir::getIntIfConstant(dataBound.getUpperbound())) {
    lb = builder.createIntegerConstant(
        loc, idxTy, *fir::getIntIfConstant(dataBound.getLowerbound()));
    ub = builder.createIntegerConstant(
        loc, idxTy, *fir::getIntIfConstant(dataBound.getUpperbound()));
    step = builder.createIntegerConstant(loc, idxTy, 1);
  } else if (dataBound.getExtent()) {
    lb = builder.createIntegerConstant(loc, idxTy, 0);
    ub = builder.createIntegerConstant(
        loc, idxTy, *fir::getIntIfConstant(dataBound.getExtent()) - 1);
    step = builder.createIntegerConstant(loc, idxTy, 1);
  } else {
    llvm::report_fatal_error("Expect constant lb/ub or extent");
  }
  return {lb, ub, step};
}

static mlir::Value genShapeFromBoundsOrArgs(
    mlir::Location loc, fir::FirOpBuilder &builder, fir::SequenceType seqTy,
    const llvm::SmallVector<mlir::Value> &bounds, mlir::ValueRange arguments) {
  llvm::SmallVector<mlir::Value> args;
  if (bounds.empty() && seqTy) {
    if (seqTy.hasDynamicExtents()) {
      assert(!arguments.empty() && "arguments must hold the entity");
      auto entity = hlfir::Entity{arguments[0]};
      return hlfir::genShape(loc, builder, entity);
    }
    return genShapeOp(builder, seqTy, loc).getResult();
  } else if (areAllBoundConstant(bounds)) {
    for (auto bound : llvm::reverse(bounds)) {
      auto dataBound =
          mlir::cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
      args.append(genConstantBounds(builder, loc, dataBound));
    }
  } else {
    assert(((arguments.size() - 2) / 3 == seqTy.getDimension()) &&
           "Expect 3 block arguments per dimension");
    for (auto arg : arguments.drop_front(2))
      args.push_back(arg);
  }

  assert(args.size() % 3 == 0 && "Triplets must be a multiple of 3");
  llvm::SmallVector<mlir::Value> extents;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  for (unsigned i = 0; i < args.size(); i += 3) {
    mlir::Value s1 =
        builder.create<mlir::arith::SubIOp>(loc, args[i + 1], args[0]);
    mlir::Value s2 = builder.create<mlir::arith::AddIOp>(loc, s1, one);
    mlir::Value s3 = builder.create<mlir::arith::DivSIOp>(loc, s2, args[i + 2]);
    mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, s3, zero);
    mlir::Value ext = builder.create<mlir::arith::SelectOp>(loc, cmp, s3, zero);
    extents.push_back(ext);
  }
  return builder.create<fir::ShapeOp>(loc, extents);
}

static hlfir::DesignateOp::Subscripts
getSubscriptsFromArgs(mlir::ValueRange args) {
  hlfir::DesignateOp::Subscripts triplets;
  for (unsigned i = 2; i < args.size(); i += 3)
    triplets.emplace_back(
        hlfir::DesignateOp::Triplet{args[i], args[i + 1], args[i + 2]});
  return triplets;
}

static hlfir::Entity genDesignateWithTriplets(
    fir::FirOpBuilder &builder, mlir::Location loc, hlfir::Entity &entity,
    hlfir::DesignateOp::Subscripts &triplets, mlir::Value shape) {
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, entity, lenParams);
  auto designate = builder.create<hlfir::DesignateOp>(
      loc, entity.getBase().getType(), entity, /*component=*/"",
      /*componentShape=*/mlir::Value{}, triplets,
      /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt, shape,
      lenParams);
  return hlfir::Entity{designate.getResult()};
}

mlir::acc::FirstprivateRecipeOp Fortran::lower::createOrGetFirstprivateRecipe(
    fir::FirOpBuilder &builder, llvm::StringRef recipeName, mlir::Location loc,
    mlir::Type ty, llvm::SmallVector<mlir::Value> &bounds) {
  mlir::ModuleOp mod =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  if (auto recipe =
          mod.lookupSymbol<mlir::acc::FirstprivateRecipeOp>(recipeName))
    return recipe;

  auto ip = builder.saveInsertionPoint();
  auto recipe = genRecipeOp<mlir::acc::FirstprivateRecipeOp>(
      builder, mod, recipeName, loc, ty);
  bool allConstantBound = areAllBoundConstant(bounds);
  llvm::SmallVector<mlir::Type> argsTy{ty, ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc, loc};
  if (!allConstantBound) {
    for (mlir::Value bound : llvm::reverse(bounds)) {
      auto dataBound =
          mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
      argsTy.push_back(dataBound.getLowerbound().getType());
      argsLoc.push_back(dataBound.getLowerbound().getLoc());
      argsTy.push_back(dataBound.getUpperbound().getType());
      argsLoc.push_back(dataBound.getUpperbound().getLoc());
      argsTy.push_back(dataBound.getStartIdx().getType());
      argsLoc.push_back(dataBound.getStartIdx().getLoc());
    }
  }
  builder.createBlock(&recipe.getCopyRegion(), recipe.getCopyRegion().end(),
                      argsTy, argsLoc);

  builder.setInsertionPointToEnd(&recipe.getCopyRegion().back());
  ty = fir::unwrapRefType(ty);
  if (fir::isa_trivial(ty)) {
    mlir::Value initValue = builder.create<fir::LoadOp>(
        loc, recipe.getCopyRegion().front().getArgument(0));
    builder.create<fir::StoreOp>(loc, initValue,
                                 recipe.getCopyRegion().front().getArgument(1));
  } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(ty)) {
    fir::FirOpBuilder firBuilder{builder, recipe.getOperation()};
    auto shape = genShapeFromBoundsOrArgs(
        loc, firBuilder, seqTy, bounds, recipe.getCopyRegion().getArguments());

    auto leftDeclOp = builder.create<hlfir::DeclareOp>(
        loc, recipe.getCopyRegion().getArgument(0), llvm::StringRef{}, shape,
        llvm::ArrayRef<mlir::Value>{}, /*dummy_scope=*/nullptr,
        fir::FortranVariableFlagsAttr{});
    auto rightDeclOp = builder.create<hlfir::DeclareOp>(
        loc, recipe.getCopyRegion().getArgument(1), llvm::StringRef{}, shape,
        llvm::ArrayRef<mlir::Value>{}, /*dummy_scope=*/nullptr,
        fir::FortranVariableFlagsAttr{});

    hlfir::DesignateOp::Subscripts triplets =
        getSubscriptsFromArgs(recipe.getCopyRegion().getArguments());
    auto leftEntity = hlfir::Entity{leftDeclOp.getBase()};
    auto left =
        genDesignateWithTriplets(firBuilder, loc, leftEntity, triplets, shape);
    auto rightEntity = hlfir::Entity{rightDeclOp.getBase()};
    auto right =
        genDesignateWithTriplets(firBuilder, loc, rightEntity, triplets, shape);

    firBuilder.create<hlfir::AssignOp>(loc, left, right);

  } else if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(ty)) {
    fir::FirOpBuilder firBuilder{builder, recipe.getOperation()};
    llvm::SmallVector<mlir::Value> tripletArgs;
    mlir::Type innerTy = fir::extractSequenceType(boxTy);
    fir::SequenceType seqTy =
        mlir::dyn_cast_or_null<fir::SequenceType>(innerTy);
    if (!seqTy)
      TODO(loc, "Unsupported boxed type in OpenACC firstprivate");

    auto shape = genShapeFromBoundsOrArgs(
        loc, firBuilder, seqTy, bounds, recipe.getCopyRegion().getArguments());
    hlfir::DesignateOp::Subscripts triplets =
        getSubscriptsFromArgs(recipe.getCopyRegion().getArguments());
    auto leftEntity = hlfir::Entity{recipe.getCopyRegion().getArgument(0)};
    auto left =
        genDesignateWithTriplets(firBuilder, loc, leftEntity, triplets, shape);
    auto rightEntity = hlfir::Entity{recipe.getCopyRegion().getArgument(1)};
    auto right =
        genDesignateWithTriplets(firBuilder, loc, rightEntity, triplets, shape);
    firBuilder.create<hlfir::AssignOp>(loc, left, right);
  }

  builder.create<mlir::acc::TerminatorOp>(loc);
  builder.restoreInsertionPoint(ip);
  return recipe;
}

/// Get a string representation of the bounds.
std::string getBoundsString(llvm::SmallVector<mlir::Value> &bounds) {
  std::stringstream boundStr;
  if (!bounds.empty())
    boundStr << "_section_";
  llvm::interleave(
      bounds,
      [&](mlir::Value bound) {
        auto boundsOp =
            mlir::cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
        if (boundsOp.getLowerbound() &&
            fir::getIntIfConstant(boundsOp.getLowerbound()) &&
            boundsOp.getUpperbound() &&
            fir::getIntIfConstant(boundsOp.getUpperbound())) {
          boundStr << "lb" << *fir::getIntIfConstant(boundsOp.getLowerbound())
                   << ".ub" << *fir::getIntIfConstant(boundsOp.getUpperbound());
        } else if (boundsOp.getExtent() &&
                   fir::getIntIfConstant(boundsOp.getExtent())) {
          boundStr << "ext" << *fir::getIntIfConstant(boundsOp.getExtent());
        } else {
          boundStr << "?";
        }
      },
      [&] { boundStr << "x"; });
  return boundStr.str();
}

/// Rebuild the array type from the acc.bounds operation with constant
/// lowerbound/upperbound or extent.
mlir::Type getTypeFromBounds(llvm::SmallVector<mlir::Value> &bounds,
                             mlir::Type ty) {
  auto seqTy =
      mlir::dyn_cast_or_null<fir::SequenceType>(fir::unwrapRefType(ty));
  if (!bounds.empty() && seqTy) {
    llvm::SmallVector<int64_t> shape;
    for (auto b : bounds) {
      auto boundsOp =
          mlir::dyn_cast<mlir::acc::DataBoundsOp>(b.getDefiningOp());
      if (boundsOp.getLowerbound() &&
          fir::getIntIfConstant(boundsOp.getLowerbound()) &&
          boundsOp.getUpperbound() &&
          fir::getIntIfConstant(boundsOp.getUpperbound())) {
        int64_t ext = *fir::getIntIfConstant(boundsOp.getUpperbound()) -
                      *fir::getIntIfConstant(boundsOp.getLowerbound()) + 1;
        shape.push_back(ext);
      } else if (boundsOp.getExtent() &&
                 fir::getIntIfConstant(boundsOp.getExtent())) {
        shape.push_back(*fir::getIntIfConstant(boundsOp.getExtent()));
      } else {
        return ty; // TODO: handle dynamic shaped array slice.
      }
    }
    if (shape.empty() || shape.size() != bounds.size())
      return ty;
    auto newSeqTy = fir::SequenceType::get(shape, seqTy.getEleTy());
    if (mlir::isa<fir::ReferenceType, fir::PointerType>(ty))
      return fir::ReferenceType::get(newSeqTy);
    return newSeqTy;
  }
  return ty;
}

template <typename RecipeOp>
static void genPrivatizationRecipes(
    const Fortran::parser::AccObjectList &objectList,
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    llvm::SmallVectorImpl<mlir::Value> &dataOperands,
    llvm::SmallVector<mlir::Attribute> &privatizationRecipes,
    llvm::ArrayRef<mlir::Value> async,
    llvm::ArrayRef<mlir::Attribute> asyncDeviceTypes,
    llvm::ArrayRef<mlir::Attribute> asyncOnlyDeviceTypes) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::evaluate::ExpressionAnalyzer ea{semanticsContext};
  for (const auto &accObject : objectList.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    Fortran::semantics::Symbol &symbol = getSymbolFromAccObject(accObject);
    Fortran::semantics::MaybeExpr designator = Fortran::common::visit(
        [&](auto &&s) { return ea.Analyze(s); }, accObject.u);
    fir::factory::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            mlir::acc::DataBoundsOp, mlir::acc::DataBoundsType>(
            converter, builder, semanticsContext, stmtCtx, symbol, designator,
            operandLocation, asFortran, bounds,
            /*treatIndexAsSection=*/true, /*unwrapFirBox=*/unwrapFirBox,
            /*genDefaultBounds=*/generateDefaultBounds,
            /*strideIncludeLowerExtent=*/strideIncludeLowerExtent);
    LLVM_DEBUG(llvm::dbgs() << __func__ << "\n"; info.dump(llvm::dbgs()));

    RecipeOp recipe;
    mlir::Type retTy = getTypeFromBounds(bounds, info.addr.getType());
    if constexpr (std::is_same_v<RecipeOp, mlir::acc::PrivateRecipeOp>) {
      std::string recipeName =
          fir::getTypeAsString(retTy, converter.getKindMap(),
                               Fortran::lower::privatizationRecipePrefix);
      recipe = Fortran::lower::createOrGetPrivateRecipe(builder, recipeName,
                                                        operandLocation, retTy);
      auto op = createDataEntryOp<mlir::acc::PrivateOp>(
          builder, operandLocation, info.addr, asFortran, bounds, true,
          /*implicit=*/false, mlir::acc::DataClause::acc_private, retTy, async,
          asyncDeviceTypes, asyncOnlyDeviceTypes, /*unwrapBoxAddr=*/true);
      dataOperands.push_back(op.getAccVar());
    } else {
      std::string suffix =
          areAllBoundConstant(bounds) ? getBoundsString(bounds) : "";
      std::string recipeName = fir::getTypeAsString(
          retTy, converter.getKindMap(), "firstprivatization" + suffix);
      recipe = Fortran::lower::createOrGetFirstprivateRecipe(
          builder, recipeName, operandLocation, retTy, bounds);
      auto op = createDataEntryOp<mlir::acc::FirstprivateOp>(
          builder, operandLocation, info.addr, asFortran, bounds, true,
          /*implicit=*/false, mlir::acc::DataClause::acc_firstprivate, retTy,
          async, asyncDeviceTypes, asyncOnlyDeviceTypes,
          /*unwrapBoxAddr=*/true);
      dataOperands.push_back(op.getAccVar());
    }
    privatizationRecipes.push_back(mlir::SymbolRefAttr::get(
        builder.getContext(), recipe.getSymName().str()));
  }
}

/// Return the corresponding enum value for the mlir::acc::ReductionOperator
/// from the parser representation.
static mlir::acc::ReductionOperator
getReductionOperator(const Fortran::parser::ReductionOperator &op) {
  switch (op.v) {
  case Fortran::parser::ReductionOperator::Operator::Plus:
    return mlir::acc::ReductionOperator::AccAdd;
  case Fortran::parser::ReductionOperator::Operator::Multiply:
    return mlir::acc::ReductionOperator::AccMul;
  case Fortran::parser::ReductionOperator::Operator::Max:
    return mlir::acc::ReductionOperator::AccMax;
  case Fortran::parser::ReductionOperator::Operator::Min:
    return mlir::acc::ReductionOperator::AccMin;
  case Fortran::parser::ReductionOperator::Operator::Iand:
    return mlir::acc::ReductionOperator::AccIand;
  case Fortran::parser::ReductionOperator::Operator::Ior:
    return mlir::acc::ReductionOperator::AccIor;
  case Fortran::parser::ReductionOperator::Operator::Ieor:
    return mlir::acc::ReductionOperator::AccXor;
  case Fortran::parser::ReductionOperator::Operator::And:
    return mlir::acc::ReductionOperator::AccLand;
  case Fortran::parser::ReductionOperator::Operator::Or:
    return mlir::acc::ReductionOperator::AccLor;
  case Fortran::parser::ReductionOperator::Operator::Eqv:
    return mlir::acc::ReductionOperator::AccEqv;
  case Fortran::parser::ReductionOperator::Operator::Neqv:
    return mlir::acc::ReductionOperator::AccNeqv;
  }
  llvm_unreachable("unexpected reduction operator");
}

template <typename Op>
static mlir::Value genLogicalCombiner(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value value1,
                                      mlir::Value value2) {
  mlir::Type i1 = builder.getI1Type();
  mlir::Value v1 = builder.create<fir::ConvertOp>(loc, i1, value1);
  mlir::Value v2 = builder.create<fir::ConvertOp>(loc, i1, value2);
  mlir::Value combined = builder.create<Op>(loc, v1, v2);
  return builder.create<fir::ConvertOp>(loc, value1.getType(), combined);
}

static mlir::Value genComparisonCombiner(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::arith::CmpIPredicate pred,
                                         mlir::Value value1,
                                         mlir::Value value2) {
  mlir::Type i1 = builder.getI1Type();
  mlir::Value v1 = builder.create<fir::ConvertOp>(loc, i1, value1);
  mlir::Value v2 = builder.create<fir::ConvertOp>(loc, i1, value2);
  mlir::Value add = builder.create<mlir::arith::CmpIOp>(loc, pred, v1, v2);
  return builder.create<fir::ConvertOp>(loc, value1.getType(), add);
}

static mlir::Value genScalarCombiner(fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::acc::ReductionOperator op,
                                     mlir::Type ty, mlir::Value value1,
                                     mlir::Value value2) {
  value1 = builder.loadIfRef(loc, value1);
  value2 = builder.loadIfRef(loc, value2);
  if (op == mlir::acc::ReductionOperator::AccAdd) {
    if (ty.isIntOrIndex())
      return builder.create<mlir::arith::AddIOp>(loc, value1, value2);
    if (mlir::isa<mlir::FloatType>(ty))
      return builder.create<mlir::arith::AddFOp>(loc, value1, value2);
    if (auto cmplxTy = mlir::dyn_cast_or_null<mlir::ComplexType>(ty))
      return builder.create<fir::AddcOp>(loc, value1, value2);
    TODO(loc, "reduction add type");
  }

  if (op == mlir::acc::ReductionOperator::AccMul) {
    if (ty.isIntOrIndex())
      return builder.create<mlir::arith::MulIOp>(loc, value1, value2);
    if (mlir::isa<mlir::FloatType>(ty))
      return builder.create<mlir::arith::MulFOp>(loc, value1, value2);
    if (mlir::isa<mlir::ComplexType>(ty))
      return builder.create<fir::MulcOp>(loc, value1, value2);
    TODO(loc, "reduction mul type");
  }

  if (op == mlir::acc::ReductionOperator::AccMin)
    return fir::genMin(builder, loc, {value1, value2});

  if (op == mlir::acc::ReductionOperator::AccMax)
    return fir::genMax(builder, loc, {value1, value2});

  if (op == mlir::acc::ReductionOperator::AccIand)
    return builder.create<mlir::arith::AndIOp>(loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccIor)
    return builder.create<mlir::arith::OrIOp>(loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccXor)
    return builder.create<mlir::arith::XOrIOp>(loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccLand)
    return genLogicalCombiner<mlir::arith::AndIOp>(builder, loc, value1,
                                                   value2);

  if (op == mlir::acc::ReductionOperator::AccLor)
    return genLogicalCombiner<mlir::arith::OrIOp>(builder, loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccEqv)
    return genComparisonCombiner(builder, loc, mlir::arith::CmpIPredicate::eq,
                                 value1, value2);

  if (op == mlir::acc::ReductionOperator::AccNeqv)
    return genComparisonCombiner(builder, loc, mlir::arith::CmpIPredicate::ne,
                                 value1, value2);

  TODO(loc, "reduction operator");
}

static hlfir::DesignateOp::Subscripts
getTripletsFromArgs(mlir::acc::ReductionRecipeOp recipe) {
  hlfir::DesignateOp::Subscripts triplets;
  for (unsigned i = 2; i < recipe.getCombinerRegion().getArguments().size();
       i += 3)
    triplets.emplace_back(hlfir::DesignateOp::Triplet{
        recipe.getCombinerRegion().getArgument(i),
        recipe.getCombinerRegion().getArgument(i + 1),
        recipe.getCombinerRegion().getArgument(i + 2)});
  return triplets;
}

static void genCombiner(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::acc::ReductionOperator op, mlir::Type ty,
                        mlir::Value value1, mlir::Value value2,
                        mlir::acc::ReductionRecipeOp &recipe,
                        llvm::SmallVector<mlir::Value> &bounds,
                        bool allConstantBound) {
  ty = fir::unwrapRefType(ty);

  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty)) {
    mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
    llvm::SmallVector<fir::DoLoopOp> loops;
    llvm::SmallVector<mlir::Value> ivs;
    if (seqTy.hasDynamicExtents()) {
      auto shape =
          genShapeFromBoundsOrArgs(loc, builder, seqTy, bounds,
                                   recipe.getCombinerRegion().getArguments());
      auto v1DeclareOp = builder.create<hlfir::DeclareOp>(
          loc, value1, llvm::StringRef{}, shape, llvm::ArrayRef<mlir::Value>{},
          /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});
      auto v2DeclareOp = builder.create<hlfir::DeclareOp>(
          loc, value2, llvm::StringRef{}, shape, llvm::ArrayRef<mlir::Value>{},
          /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});
      hlfir::DesignateOp::Subscripts triplets = getTripletsFromArgs(recipe);

      llvm::SmallVector<mlir::Value> lenParamsLeft;
      auto leftEntity = hlfir::Entity{v1DeclareOp.getBase()};
      hlfir::genLengthParameters(loc, builder, leftEntity, lenParamsLeft);
      auto leftDesignate = builder.create<hlfir::DesignateOp>(
          loc, v1DeclareOp.getBase().getType(), v1DeclareOp.getBase(),
          /*component=*/"",
          /*componentShape=*/mlir::Value{}, triplets,
          /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt,
          shape, lenParamsLeft);
      auto left = hlfir::Entity{leftDesignate.getResult()};

      llvm::SmallVector<mlir::Value> lenParamsRight;
      auto rightEntity = hlfir::Entity{v2DeclareOp.getBase()};
      hlfir::genLengthParameters(loc, builder, rightEntity, lenParamsLeft);
      auto rightDesignate = builder.create<hlfir::DesignateOp>(
          loc, v2DeclareOp.getBase().getType(), v2DeclareOp.getBase(),
          /*component=*/"",
          /*componentShape=*/mlir::Value{}, triplets,
          /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt,
          shape, lenParamsRight);
      auto right = hlfir::Entity{rightDesignate.getResult()};

      llvm::SmallVector<mlir::Value, 1> typeParams;
      auto genKernel = [&builder, &loc, op, seqTy, &left, &right](
                           mlir::Location l, fir::FirOpBuilder &b,
                           mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
        auto leftElement = hlfir::getElementAt(l, b, left, oneBasedIndices);
        auto rightElement = hlfir::getElementAt(l, b, right, oneBasedIndices);
        auto leftVal = hlfir::loadTrivialScalar(l, b, leftElement);
        auto rightVal = hlfir::loadTrivialScalar(l, b, rightElement);
        return hlfir::Entity{genScalarCombiner(
            builder, loc, op, seqTy.getEleTy(), leftVal, rightVal)};
      };
      mlir::Value elemental = hlfir::genElementalOp(
          loc, builder, seqTy.getEleTy(), shape, typeParams, genKernel,
          /*isUnordered=*/true);
      builder.create<hlfir::AssignOp>(loc, elemental, v1DeclareOp.getBase());
      return;
    }
    if (bounds.empty()) {
      llvm::SmallVector<mlir::Value> extents;
      mlir::Type idxTy = builder.getIndexType();
      for (auto extent : seqTy.getShape()) {
        mlir::Value lb = builder.create<mlir::arith::ConstantOp>(
            loc, idxTy, builder.getIntegerAttr(idxTy, 0));
        mlir::Value ub = builder.create<mlir::arith::ConstantOp>(
            loc, idxTy, builder.getIntegerAttr(idxTy, extent - 1));
        mlir::Value step = builder.create<mlir::arith::ConstantOp>(
            loc, idxTy, builder.getIntegerAttr(idxTy, 1));
        auto loop = builder.create<fir::DoLoopOp>(loc, lb, ub, step,
                                                  /*unordered=*/false);
        builder.setInsertionPointToStart(loop.getBody());
        loops.push_back(loop);
        ivs.push_back(loop.getInductionVar());
      }
    } else if (allConstantBound) {
      // Use the constant bound directly in the combiner region so they do not
      // need to be passed as block argument.
      assert(!bounds.empty() &&
             "seq type with constant bounds cannot have empty bounds");
      for (auto bound : llvm::reverse(bounds)) {
        auto dataBound =
            mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
        llvm::SmallVector<mlir::Value> values =
            genConstantBounds(builder, loc, dataBound);
        auto loop =
            builder.create<fir::DoLoopOp>(loc, values[0], values[1], values[2],
                                          /*unordered=*/false);
        builder.setInsertionPointToStart(loop.getBody());
        loops.push_back(loop);
        ivs.push_back(loop.getInductionVar());
      }
    } else {
      // Lowerbound, upperbound and step are passed as block arguments.
      [[maybe_unused]] unsigned nbRangeArgs =
          recipe.getCombinerRegion().getArguments().size() - 2;
      assert((nbRangeArgs / 3 == seqTy.getDimension()) &&
             "Expect 3 block arguments per dimension");
      for (unsigned i = 2; i < recipe.getCombinerRegion().getArguments().size();
           i += 3) {
        mlir::Value lb = recipe.getCombinerRegion().getArgument(i);
        mlir::Value ub = recipe.getCombinerRegion().getArgument(i + 1);
        mlir::Value step = recipe.getCombinerRegion().getArgument(i + 2);
        auto loop = builder.create<fir::DoLoopOp>(loc, lb, ub, step,
                                                  /*unordered=*/false);
        builder.setInsertionPointToStart(loop.getBody());
        loops.push_back(loop);
        ivs.push_back(loop.getInductionVar());
      }
    }
    auto addr1 = builder.create<fir::CoordinateOp>(loc, refTy, value1, ivs);
    auto addr2 = builder.create<fir::CoordinateOp>(loc, refTy, value2, ivs);
    auto load1 = builder.create<fir::LoadOp>(loc, addr1);
    auto load2 = builder.create<fir::LoadOp>(loc, addr2);
    mlir::Value res =
        genScalarCombiner(builder, loc, op, seqTy.getEleTy(), load1, load2);
    builder.create<fir::StoreOp>(loc, res, addr1);
    builder.setInsertionPointAfter(loops[0]);
  } else if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty)) {
    mlir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
    if (fir::isa_trivial(innerTy)) {
      mlir::Value boxAddr1 = value1, boxAddr2 = value2;
      if (fir::isBoxAddress(boxAddr1.getType()))
        boxAddr1 = builder.create<fir::LoadOp>(loc, boxAddr1);
      if (fir::isBoxAddress(boxAddr2.getType()))
        boxAddr2 = builder.create<fir::LoadOp>(loc, boxAddr2);
      boxAddr1 = builder.create<fir::BoxAddrOp>(loc, boxAddr1);
      boxAddr2 = builder.create<fir::BoxAddrOp>(loc, boxAddr2);
      auto leftEntity = hlfir::Entity{boxAddr1};
      auto rightEntity = hlfir::Entity{boxAddr2};

      auto leftVal = hlfir::loadTrivialScalar(loc, builder, leftEntity);
      auto rightVal = hlfir::loadTrivialScalar(loc, builder, rightEntity);
      mlir::Value res =
          genScalarCombiner(builder, loc, op, innerTy, leftVal, rightVal);
      builder.create<hlfir::AssignOp>(loc, res, boxAddr1);
    } else {
      mlir::Type innerTy = fir::extractSequenceType(boxTy);
      fir::SequenceType seqTy =
          mlir::dyn_cast_or_null<fir::SequenceType>(innerTy);
      if (!seqTy)
        TODO(loc, "Unsupported boxed type in OpenACC reduction combiner");

      auto shape =
          genShapeFromBoundsOrArgs(loc, builder, seqTy, bounds,
                                   recipe.getCombinerRegion().getArguments());
      hlfir::DesignateOp::Subscripts triplets =
          getSubscriptsFromArgs(recipe.getCombinerRegion().getArguments());
      auto leftEntity = hlfir::Entity{value1};
      if (fir::isBoxAddress(value1.getType()))
        leftEntity =
            hlfir::Entity{builder.create<fir::LoadOp>(loc, value1).getResult()};
      auto left =
          genDesignateWithTriplets(builder, loc, leftEntity, triplets, shape);
      auto rightEntity = hlfir::Entity{value2};
      if (fir::isBoxAddress(value2.getType()))
        rightEntity =
            hlfir::Entity{builder.create<fir::LoadOp>(loc, value2).getResult()};
      auto right =
          genDesignateWithTriplets(builder, loc, rightEntity, triplets, shape);

      llvm::SmallVector<mlir::Value, 1> typeParams;
      auto genKernel = [&builder, &loc, op, seqTy, &left, &right](
                           mlir::Location l, fir::FirOpBuilder &b,
                           mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
        auto leftElement = hlfir::getElementAt(l, b, left, oneBasedIndices);
        auto rightElement = hlfir::getElementAt(l, b, right, oneBasedIndices);
        auto leftVal = hlfir::loadTrivialScalar(l, b, leftElement);
        auto rightVal = hlfir::loadTrivialScalar(l, b, rightElement);
        return hlfir::Entity{genScalarCombiner(
            builder, loc, op, seqTy.getEleTy(), leftVal, rightVal)};
      };
      mlir::Value elemental = hlfir::genElementalOp(
          loc, builder, seqTy.getEleTy(), shape, typeParams, genKernel,
          /*isUnordered=*/true);
      builder.create<hlfir::AssignOp>(loc, elemental, value1);
    }
  } else {
    mlir::Value res = genScalarCombiner(builder, loc, op, ty, value1, value2);
    builder.create<fir::StoreOp>(loc, res, value1);
  }
}

mlir::acc::ReductionRecipeOp Fortran::lower::createOrGetReductionRecipe(
    fir::FirOpBuilder &builder, llvm::StringRef recipeName, mlir::Location loc,
    mlir::Type ty, mlir::acc::ReductionOperator op,
    llvm::SmallVector<mlir::Value> &bounds) {
  mlir::ModuleOp mod =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  if (auto recipe = mod.lookupSymbol<mlir::acc::ReductionRecipeOp>(recipeName))
    return recipe;

  auto ip = builder.saveInsertionPoint();

  auto recipe = genRecipeOp<mlir::acc::ReductionRecipeOp>(
      builder, mod, recipeName, loc, ty, op);

  // The two first block arguments are the two values to be combined.
  // The next arguments are the iteration ranges (lb, ub, step) to be used
  // for the combiner if needed.
  llvm::SmallVector<mlir::Type> argsTy{ty, ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc, loc};
  bool allConstantBound = areAllBoundConstant(bounds);
  if (!allConstantBound) {
    for (mlir::Value bound : llvm::reverse(bounds)) {
      auto dataBound =
          mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
      argsTy.push_back(dataBound.getLowerbound().getType());
      argsLoc.push_back(dataBound.getLowerbound().getLoc());
      argsTy.push_back(dataBound.getUpperbound().getType());
      argsLoc.push_back(dataBound.getUpperbound().getLoc());
      argsTy.push_back(dataBound.getStartIdx().getType());
      argsLoc.push_back(dataBound.getStartIdx().getLoc());
    }
  }
  builder.createBlock(&recipe.getCombinerRegion(),
                      recipe.getCombinerRegion().end(), argsTy, argsLoc);
  builder.setInsertionPointToEnd(&recipe.getCombinerRegion().back());
  mlir::Value v1 = recipe.getCombinerRegion().front().getArgument(0);
  mlir::Value v2 = recipe.getCombinerRegion().front().getArgument(1);
  genCombiner(builder, loc, op, ty, v1, v2, recipe, bounds, allConstantBound);
  builder.create<mlir::acc::YieldOp>(loc, v1);
  builder.restoreInsertionPoint(ip);
  return recipe;
}

static bool isSupportedReductionType(mlir::Type ty) {
  ty = fir::unwrapRefType(ty);
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty))
    return isSupportedReductionType(boxTy.getEleTy());
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
    return isSupportedReductionType(seqTy.getEleTy());
  if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
    return isSupportedReductionType(heapTy.getEleTy());
  if (auto ptrTy = mlir::dyn_cast<fir::PointerType>(ty))
    return isSupportedReductionType(ptrTy.getEleTy());
  return fir::isa_trivial(ty);
}

static void
genReductions(const Fortran::parser::AccObjectListWithReduction &objectList,
              Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semanticsContext,
              Fortran::lower::StatementContext &stmtCtx,
              llvm::SmallVectorImpl<mlir::Value> &reductionOperands,
              llvm::SmallVector<mlir::Attribute> &reductionRecipes,
              llvm::ArrayRef<mlir::Value> async,
              llvm::ArrayRef<mlir::Attribute> asyncDeviceTypes,
              llvm::ArrayRef<mlir::Attribute> asyncOnlyDeviceTypes) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const auto &objects = std::get<Fortran::parser::AccObjectList>(objectList.t);
  const auto &op = std::get<Fortran::parser::ReductionOperator>(objectList.t);
  mlir::acc::ReductionOperator mlirOp = getReductionOperator(op);
  Fortran::evaluate::ExpressionAnalyzer ea{semanticsContext};
  for (const auto &accObject : objects.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    Fortran::semantics::Symbol &symbol = getSymbolFromAccObject(accObject);
    Fortran::semantics::MaybeExpr designator = Fortran::common::visit(
        [&](auto &&s) { return ea.Analyze(s); }, accObject.u);
    fir::factory::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            mlir::acc::DataBoundsOp, mlir::acc::DataBoundsType>(
            converter, builder, semanticsContext, stmtCtx, symbol, designator,
            operandLocation, asFortran, bounds,
            /*treatIndexAsSection=*/true, /*unwrapFirBox=*/unwrapFirBox,
            /*genDefaultBounds=*/generateDefaultBounds,
            /*strideIncludeLowerExtent=*/strideIncludeLowerExtent);
    LLVM_DEBUG(llvm::dbgs() << __func__ << "\n"; info.dump(llvm::dbgs()));

    mlir::Type reductionTy = fir::unwrapRefType(info.addr.getType());
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(reductionTy))
      reductionTy = seqTy.getEleTy();

    if (!isSupportedReductionType(reductionTy))
      TODO(operandLocation, "reduction with unsupported type");

    auto op = createDataEntryOp<mlir::acc::ReductionOp>(
        builder, operandLocation, info.addr, asFortran, bounds,
        /*structured=*/true, /*implicit=*/false,
        mlir::acc::DataClause::acc_reduction, info.addr.getType(), async,
        asyncDeviceTypes, asyncOnlyDeviceTypes, /*unwrapBoxAddr=*/true);
    mlir::Type ty = op.getAccVar().getType();
    if (!areAllBoundConstant(bounds) ||
        fir::isAssumedShape(info.addr.getType()) ||
        fir::isAllocatableOrPointerArray(info.addr.getType()))
      ty = info.addr.getType();
    std::string suffix =
        areAllBoundConstant(bounds) ? getBoundsString(bounds) : "";
    std::string recipeName = fir::getTypeAsString(
        ty, converter.getKindMap(),
        ("reduction_" + stringifyReductionOperator(mlirOp)).str() + suffix);

    mlir::acc::ReductionRecipeOp recipe =
        Fortran::lower::createOrGetReductionRecipe(
            builder, recipeName, operandLocation, ty, mlirOp, bounds);
    reductionRecipes.push_back(mlir::SymbolRefAttr::get(
        builder.getContext(), recipe.getSymName().str()));
    reductionOperands.push_back(op.getAccVar());
  }
}

template <typename Op, typename Terminator>
static Op
createRegionOp(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Location returnLoc, Fortran::lower::pft::Evaluation &eval,
               const llvm::SmallVectorImpl<mlir::Value> &operands,
               const llvm::SmallVectorImpl<int32_t> &operandSegments,
               bool outerCombined = false,
               llvm::SmallVector<mlir::Type> retTy = {},
               mlir::Value yieldValue = {}, mlir::TypeRange argsTy = {},
               llvm::SmallVector<mlir::Location> locs = {}) {
  Op op = builder.create<Op>(loc, retTy, operands);
  builder.createBlock(&op.getRegion(), op.getRegion().end(), argsTy, locs);
  mlir::Block &block = op.getRegion().back();
  builder.setInsertionPointToStart(&block);

  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getDenseI32ArrayAttr(operandSegments));

  // Place the insertion point to the start of the first block.
  builder.setInsertionPointToStart(&block);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (eval.lowerAsUnstructured() && !outerCombined)
    Fortran::lower::createEmptyRegionBlocks<mlir::acc::TerminatorOp,
                                            mlir::acc::YieldOp>(
        builder, eval.getNestedEvaluations());

  if (yieldValue) {
    if constexpr (std::is_same_v<Terminator, mlir::acc::YieldOp>) {
      Terminator yieldOp = builder.create<Terminator>(returnLoc, yieldValue);
      yieldValue.getDefiningOp()->moveBefore(yieldOp);
    } else {
      builder.create<Terminator>(returnLoc);
    }
  } else {
    builder.create<Terminator>(returnLoc);
  }
  builder.setInsertionPointToStart(&block);
  return op;
}

static void genAsyncClause(Fortran::lower::AbstractConverter &converter,
                           const Fortran::parser::AccClause::Async *asyncClause,
                           mlir::Value &async, bool &addAsyncAttr,
                           Fortran::lower::StatementContext &stmtCtx) {
  const auto &asyncClauseValue = asyncClause->v;
  if (asyncClauseValue) { // async has a value.
    async = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(*asyncClauseValue), stmtCtx));
  } else {
    addAsyncAttr = true;
  }
}

static void
genAsyncClause(Fortran::lower::AbstractConverter &converter,
               const Fortran::parser::AccClause::Async *asyncClause,
               llvm::SmallVector<mlir::Value> &async,
               llvm::SmallVector<mlir::Attribute> &asyncDeviceTypes,
               llvm::SmallVector<mlir::Attribute> &asyncOnlyDeviceTypes,
               llvm::SmallVector<mlir::Attribute> &deviceTypeAttrs,
               Fortran::lower::StatementContext &stmtCtx) {
  const auto &asyncClauseValue = asyncClause->v;
  if (asyncClauseValue) { // async has a value.
    mlir::Value asyncValue = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(*asyncClauseValue), stmtCtx));
    for (auto deviceTypeAttr : deviceTypeAttrs) {
      async.push_back(asyncValue);
      asyncDeviceTypes.push_back(deviceTypeAttr);
    }
  } else {
    for (auto deviceTypeAttr : deviceTypeAttrs)
      asyncOnlyDeviceTypes.push_back(deviceTypeAttr);
  }
}

static mlir::acc::DeviceType
getDeviceType(Fortran::common::OpenACCDeviceType device) {
  switch (device) {
  case Fortran::common::OpenACCDeviceType::Star:
    return mlir::acc::DeviceType::Star;
  case Fortran::common::OpenACCDeviceType::Default:
    return mlir::acc::DeviceType::Default;
  case Fortran::common::OpenACCDeviceType::Nvidia:
    return mlir::acc::DeviceType::Nvidia;
  case Fortran::common::OpenACCDeviceType::Radeon:
    return mlir::acc::DeviceType::Radeon;
  case Fortran::common::OpenACCDeviceType::Host:
    return mlir::acc::DeviceType::Host;
  case Fortran::common::OpenACCDeviceType::Multicore:
    return mlir::acc::DeviceType::Multicore;
  case Fortran::common::OpenACCDeviceType::None:
    return mlir::acc::DeviceType::None;
  }
  return mlir::acc::DeviceType::None;
}

static void gatherDeviceTypeAttrs(
    fir::FirOpBuilder &builder,
    const Fortran::parser::AccClause::DeviceType *deviceTypeClause,
    llvm::SmallVector<mlir::Attribute> &deviceTypes) {
  const Fortran::parser::AccDeviceTypeExprList &deviceTypeExprList =
      deviceTypeClause->v;
  for (const auto &deviceTypeExpr : deviceTypeExprList.v)
    deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
        builder.getContext(), getDeviceType(deviceTypeExpr.v)));
}

static void genIfClause(Fortran::lower::AbstractConverter &converter,
                        mlir::Location clauseLocation,
                        const Fortran::parser::AccClause::If *ifClause,
                        mlir::Value &ifCond,
                        Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Value cond = fir::getBase(converter.genExprValue(
      *Fortran::semantics::GetExpr(ifClause->v), stmtCtx, &clauseLocation));
  ifCond = firOpBuilder.createConvert(clauseLocation, firOpBuilder.getI1Type(),
                                      cond);
}

static void genWaitClause(Fortran::lower::AbstractConverter &converter,
                          const Fortran::parser::AccClause::Wait *waitClause,
                          llvm::SmallVectorImpl<mlir::Value> &operands,
                          mlir::Value &waitDevnum, bool &addWaitAttr,
                          Fortran::lower::StatementContext &stmtCtx) {
  const auto &waitClauseValue = waitClause->v;
  if (waitClauseValue) { // wait has a value.
    const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
    const auto &waitList =
        std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    for (const Fortran::parser::ScalarIntExpr &value : waitList) {
      mlir::Value v = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(value), stmtCtx));
      operands.push_back(v);
    }

    const auto &waitDevnumValue =
        std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    if (waitDevnumValue)
      waitDevnum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(*waitDevnumValue), stmtCtx));
  } else {
    addWaitAttr = true;
  }
}

static void genWaitClauseWithDeviceType(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::AccClause::Wait *waitClause,
    llvm::SmallVector<mlir::Value> &waitOperands,
    llvm::SmallVector<mlir::Attribute> &waitOperandsDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &waitOnlyDeviceTypes,
    llvm::SmallVector<bool> &hasDevnums,
    llvm::SmallVector<int32_t> &waitOperandsSegments,
    llvm::SmallVector<mlir::Attribute> deviceTypeAttrs,
    Fortran::lower::StatementContext &stmtCtx) {
  const auto &waitClauseValue = waitClause->v;
  if (waitClauseValue) { // wait has a value.
    llvm::SmallVector<mlir::Value> waitValues;

    const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
    const auto &waitDevnumValue =
        std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    bool hasDevnum = false;
    if (waitDevnumValue) {
      waitValues.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(*waitDevnumValue), stmtCtx)));
      hasDevnum = true;
    }

    const auto &waitList =
        std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    for (const Fortran::parser::ScalarIntExpr &value : waitList) {
      waitValues.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(value), stmtCtx)));
    }

    for (auto deviceTypeAttr : deviceTypeAttrs) {
      for (auto value : waitValues)
        waitOperands.push_back(value);
      waitOperandsDeviceTypes.push_back(deviceTypeAttr);
      waitOperandsSegments.push_back(waitValues.size());
      hasDevnums.push_back(hasDevnum);
    }
  } else {
    for (auto deviceTypeAttr : deviceTypeAttrs)
      waitOnlyDeviceTypes.push_back(deviceTypeAttr);
  }
}

mlir::Type getTypeFromIvTypeSize(fir::FirOpBuilder &builder,
                                 const Fortran::semantics::Symbol &ivSym) {
  std::size_t ivTypeSize = ivSym.size();
  if (ivTypeSize == 0)
    llvm::report_fatal_error("unexpected induction variable size");
  // ivTypeSize is in bytes and IntegerType needs to be in bits.
  return builder.getIntegerType(ivTypeSize * 8);
}

static void
privatizeIv(Fortran::lower::AbstractConverter &converter,
            const Fortran::semantics::Symbol &sym, mlir::Location loc,
            llvm::SmallVector<mlir::Type> &ivTypes,
            llvm::SmallVector<mlir::Location> &ivLocs,
            llvm::SmallVector<mlir::Value> &privateOperands,
            llvm::SmallVector<mlir::Value> &ivPrivate,
            llvm::SmallVector<mlir::Attribute> &privatizationRecipes,
            bool isDoConcurrent = false) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Type ivTy = getTypeFromIvTypeSize(builder, sym);
  ivTypes.push_back(ivTy);
  ivLocs.push_back(loc);
  mlir::Value ivValue = converter.getSymbolAddress(sym);
  if (!ivValue && isDoConcurrent) {
    // DO CONCURRENT induction variables are not mapped yet since they are local
    // to the DO CONCURRENT scope.
    mlir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(builder.getAllocaBlock());
    ivValue = builder.createTemporaryAlloc(loc, ivTy, toStringRef(sym.name()));
    builder.restoreInsertionPoint(insPt);
  }

  mlir::Operation *privateOp = nullptr;
  for (auto privateVal : privateOperands) {
    if (mlir::acc::getVar(privateVal.getDefiningOp()) == ivValue) {
      privateOp = privateVal.getDefiningOp();
      break;
    }
  }

  if (privateOp == nullptr) {
    std::string recipeName =
        fir::getTypeAsString(ivValue.getType(), converter.getKindMap(),
                             Fortran::lower::privatizationRecipePrefix);
    auto recipe = Fortran::lower::createOrGetPrivateRecipe(
        builder, recipeName, loc, ivValue.getType());

    std::stringstream asFortran;
    asFortran << Fortran::lower::mangle::demangleName(toStringRef(sym.name()));
    auto op = createDataEntryOp<mlir::acc::PrivateOp>(
        builder, loc, ivValue, asFortran, {}, true, /*implicit=*/true,
        mlir::acc::DataClause::acc_private, ivValue.getType(),
        /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
    privateOp = op.getOperation();

    privateOperands.push_back(op.getAccVar());
    privatizationRecipes.push_back(mlir::SymbolRefAttr::get(
        builder.getContext(), recipe.getSymName().str()));
  }

  // Map the new private iv to its symbol for the scope of the loop. bindSymbol
  // might create a hlfir.declare op, if so, we map its result in order to
  // use the sym value in the scope.
  converter.bindSymbol(sym, mlir::acc::getAccVar(privateOp));
  auto privateValue = converter.getSymbolAddress(sym);
  if (auto declareOp =
          mlir::dyn_cast<hlfir::DeclareOp>(privateValue.getDefiningOp()))
    privateValue = declareOp.getResults()[0];
  ivPrivate.push_back(privateValue);
}

static void determineDefaultLoopParMode(
    Fortran::lower::AbstractConverter &converter, mlir::acc::LoopOp &loopOp,
    llvm::SmallVector<mlir::Attribute> &seqDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &independentDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &autoDeviceTypes) {
  auto hasDeviceNone = [](mlir::Attribute attr) -> bool {
    return mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr).getValue() ==
           mlir::acc::DeviceType::None;
  };
  bool hasDefaultSeq = llvm::any_of(seqDeviceTypes, hasDeviceNone);
  bool hasDefaultIndependent =
      llvm::any_of(independentDeviceTypes, hasDeviceNone);
  bool hasDefaultAuto = llvm::any_of(autoDeviceTypes, hasDeviceNone);
  if (hasDefaultSeq || hasDefaultIndependent || hasDefaultAuto)
    return; // Default loop par mode is already specified.

  mlir::Region *currentRegion =
      converter.getFirOpBuilder().getBlock()->getParent();
  mlir::Operation *parentOp = mlir::acc::getEnclosingComputeOp(*currentRegion);
  const bool isOrphanedLoop = !parentOp;
  if (isOrphanedLoop ||
      mlir::isa_and_present<mlir::acc::ParallelOp>(parentOp)) {
    // As per OpenACC 3.3 standard section 2.9.6 independent clause:
    // A loop construct with no auto or seq clause is treated as if it has the
    // independent clause when it is an orphaned loop construct or its parent
    // compute construct is a parallel construct.
    independentDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
        converter.getFirOpBuilder().getContext(), mlir::acc::DeviceType::None));
  } else if (mlir::isa_and_present<mlir::acc::SerialOp>(parentOp)) {
    // Serial construct implies `seq` clause on loop. However, this
    // conflicts with parallelism assignment if already set. Therefore check
    // that first.
    bool hasDefaultGangWorkerOrVector =
        loopOp.hasVector() || loopOp.getVectorValue() || loopOp.hasWorker() ||
        loopOp.getWorkerValue() || loopOp.hasGang() ||
        loopOp.getGangValue(mlir::acc::GangArgType::Num) ||
        loopOp.getGangValue(mlir::acc::GangArgType::Dim) ||
        loopOp.getGangValue(mlir::acc::GangArgType::Static);
    if (!hasDefaultGangWorkerOrVector)
      seqDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
          converter.getFirOpBuilder().getContext(),
          mlir::acc::DeviceType::None));
    // Since the loop has some parallelism assigned - we cannot assign `seq`.
    // However, the `acc.loop` verifier will check that one of seq, independent,
    // or auto is marked. Seems reasonable to mark as auto since the OpenACC
    // spec does say "If not, or if it is unable to make a determination, it
    // must treat the auto clause as if it is a seq clause, and it must
    // ignore any gang, worker, or vector clauses on the loop construct"
    else
      autoDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
          converter.getFirOpBuilder().getContext(),
          mlir::acc::DeviceType::None));
  } else {
    // As per OpenACC 3.3 standard section 2.9.7 auto clause:
    // When the parent compute construct is a kernels construct, a loop
    // construct with no independent or seq clause is treated as if it has the
    // auto clause.
    assert(mlir::isa_and_present<mlir::acc::KernelsOp>(parentOp) &&
           "Expected kernels construct");
    autoDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
        converter.getFirOpBuilder().getContext(), mlir::acc::DeviceType::None));
  }
}

static mlir::acc::LoopOp createLoopOp(
    Fortran::lower::AbstractConverter &converter,
    mlir::Location currentLocation,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::DoConstruct &outerDoConstruct,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::AccClauseList &accClauseList,
    std::optional<mlir::acc::CombinedConstructsType> combinedConstructs =
        std::nullopt,
    bool needEarlyReturnHandling = false) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Value> tileOperands, privateOperands, ivPrivate,
      reductionOperands, cacheOperands, vectorOperands, workerNumOperands,
      gangOperands, lowerbounds, upperbounds, steps;
  llvm::SmallVector<mlir::Attribute> privatizationRecipes, reductionRecipes;
  llvm::SmallVector<int32_t> tileOperandsSegments, gangOperandsSegments;
  llvm::SmallVector<int64_t> collapseValues;

  llvm::SmallVector<mlir::Attribute> gangArgTypes;
  llvm::SmallVector<mlir::Attribute> seqDeviceTypes, independentDeviceTypes,
      autoDeviceTypes, vectorOperandsDeviceTypes, workerNumOperandsDeviceTypes,
      vectorDeviceTypes, workerNumDeviceTypes, tileOperandsDeviceTypes,
      collapseDeviceTypes, gangDeviceTypes, gangOperandsDeviceTypes;

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  crtDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None));

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *gangClause =
            std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
      if (gangClause->v) {
        const Fortran::parser::AccGangArgList &x = *gangClause->v;
        mlir::SmallVector<mlir::Value> gangValues;
        mlir::SmallVector<mlir::Attribute> gangArgs;
        for (const Fortran::parser::AccGangArg &gangArg : x.v) {
          if (const auto *num =
                  std::get_if<Fortran::parser::AccGangArg::Num>(&gangArg.u)) {
            gangValues.push_back(fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(num->v), stmtCtx)));
            gangArgs.push_back(mlir::acc::GangArgTypeAttr::get(
                builder.getContext(), mlir::acc::GangArgType::Num));
          } else if (const auto *staticArg =
                         std::get_if<Fortran::parser::AccGangArg::Static>(
                             &gangArg.u)) {
            const Fortran::parser::AccSizeExpr &sizeExpr = staticArg->v;
            if (sizeExpr.v) {
              gangValues.push_back(fir::getBase(converter.genExprValue(
                  *Fortran::semantics::GetExpr(*sizeExpr.v), stmtCtx)));
            } else {
              // * was passed as value and will be represented as a special
              // constant.
              gangValues.push_back(builder.createIntegerConstant(
                  clauseLocation, builder.getIndexType(), starCst));
            }
            gangArgs.push_back(mlir::acc::GangArgTypeAttr::get(
                builder.getContext(), mlir::acc::GangArgType::Static));
          } else if (const auto *dim =
                         std::get_if<Fortran::parser::AccGangArg::Dim>(
                             &gangArg.u)) {
            gangValues.push_back(fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(dim->v), stmtCtx)));
            gangArgs.push_back(mlir::acc::GangArgTypeAttr::get(
                builder.getContext(), mlir::acc::GangArgType::Dim));
          }
        }
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          for (const auto &pair : llvm::zip(gangValues, gangArgs)) {
            gangOperands.push_back(std::get<0>(pair));
            gangArgTypes.push_back(std::get<1>(pair));
          }
          gangOperandsSegments.push_back(gangValues.size());
          gangOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      } else {
        for (auto crtDeviceTypeAttr : crtDeviceTypes)
          gangDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *workerClause =
                   std::get_if<Fortran::parser::AccClause::Worker>(&clause.u)) {
      if (workerClause->v) {
        mlir::Value workerNumValue = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*workerClause->v), stmtCtx));
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          workerNumOperands.push_back(workerNumValue);
          workerNumOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      } else {
        for (auto crtDeviceTypeAttr : crtDeviceTypes)
          workerNumDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *vectorClause =
                   std::get_if<Fortran::parser::AccClause::Vector>(&clause.u)) {
      if (vectorClause->v) {
        mlir::Value vectorValue = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*vectorClause->v), stmtCtx));
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          vectorOperands.push_back(vectorValue);
          vectorOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      } else {
        for (auto crtDeviceTypeAttr : crtDeviceTypes)
          vectorDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *tileClause =
                   std::get_if<Fortran::parser::AccClause::Tile>(&clause.u)) {
      const Fortran::parser::AccTileExprList &accTileExprList = tileClause->v;
      llvm::SmallVector<mlir::Value> tileValues;
      for (const auto &accTileExpr : accTileExprList.v) {
        const auto &expr =
            std::get<std::optional<Fortran::parser::ScalarIntConstantExpr>>(
                accTileExpr.t);
        if (expr) {
          tileValues.push_back(fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*expr), stmtCtx)));
        } else {
          // * was passed as value and will be represented as a special
          // constant.
          mlir::Value tileStar = builder.createIntegerConstant(
              clauseLocation, builder.getIntegerType(32), starCst);
          tileValues.push_back(tileStar);
        }
      }
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        for (auto value : tileValues)
          tileOperands.push_back(value);
        tileOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        tileOperandsSegments.push_back(tileValues.size());
      }
    } else if (const auto *privateClause =
                   std::get_if<Fortran::parser::AccClause::Private>(
                       &clause.u)) {
      genPrivatizationRecipes<mlir::acc::PrivateRecipeOp>(
          privateClause->v, converter, semanticsContext, stmtCtx,
          privateOperands, privatizationRecipes, /*async=*/{},
          /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
    } else if (const auto *reductionClause =
                   std::get_if<Fortran::parser::AccClause::Reduction>(
                       &clause.u)) {
      genReductions(reductionClause->v, converter, semanticsContext, stmtCtx,
                    reductionOperands, reductionRecipes, /*async=*/{},
                    /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
    } else if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        seqDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (std::get_if<Fortran::parser::AccClause::Independent>(
                   &clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        independentDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (std::get_if<Fortran::parser::AccClause::Auto>(&clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        autoDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    } else if (const auto *collapseClause =
                   std::get_if<Fortran::parser::AccClause::Collapse>(
                       &clause.u)) {
      const Fortran::parser::AccCollapseArg &arg = collapseClause->v;
      const auto &force = std::get<bool>(arg.t);
      if (force)
        TODO(clauseLocation, "OpenACC collapse force modifier");

      const auto &intExpr =
          std::get<Fortran::parser::ScalarIntConstantExpr>(arg.t);
      const auto *expr = Fortran::semantics::GetExpr(intExpr);
      const std::optional<int64_t> collapseValue =
          Fortran::evaluate::ToInt64(*expr);
      assert(collapseValue && "expect integer value for the collapse clause");

      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        collapseValues.push_back(*collapseValue);
        collapseDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    }
  }

  llvm::SmallVector<mlir::Type> ivTypes;
  llvm::SmallVector<mlir::Location> ivLocs;
  llvm::SmallVector<bool> inclusiveBounds;
  llvm::SmallVector<mlir::Location> locs;
  locs.push_back(currentLocation); // Location of the directive
  Fortran::lower::pft::Evaluation *crtEval = &eval.getFirstNestedEvaluation();
  bool isDoConcurrent = outerDoConstruct.IsDoConcurrent();
  if (isDoConcurrent) {
    locs.push_back(converter.genLocation(
        Fortran::parser::FindSourceLocation(outerDoConstruct)));
    const Fortran::parser::LoopControl *loopControl =
        &*outerDoConstruct.GetLoopControl();
    const auto &concurrent =
        std::get<Fortran::parser::LoopControl::Concurrent>(loopControl->u);
    if (!std::get<std::list<Fortran::parser::LocalitySpec>>(concurrent.t)
             .empty())
      TODO(currentLocation, "DO CONCURRENT with locality spec");

    const auto &concurrentHeader =
        std::get<Fortran::parser::ConcurrentHeader>(concurrent.t);
    const auto &controls =
        std::get<std::list<Fortran::parser::ConcurrentControl>>(
            concurrentHeader.t);
    for (const auto &control : controls) {
      lowerbounds.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(std::get<1>(control.t)), stmtCtx)));
      upperbounds.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(std::get<2>(control.t)), stmtCtx)));
      if (const auto &expr =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                  control.t))
        steps.push_back(fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*expr), stmtCtx)));
      else // If `step` is not present, assume it is `1`.
        steps.push_back(builder.createIntegerConstant(
            currentLocation, upperbounds[upperbounds.size() - 1].getType(), 1));

      const auto &name = std::get<Fortran::parser::Name>(control.t);
      privatizeIv(converter, *name.symbol, currentLocation, ivTypes, ivLocs,
                  privateOperands, ivPrivate, privatizationRecipes,
                  isDoConcurrent);

      inclusiveBounds.push_back(true);
    }
  } else {
    int64_t collapseValue = Fortran::lower::getCollapseValue(accClauseList);
    for (unsigned i = 0; i < collapseValue; ++i) {
      const Fortran::parser::LoopControl *loopControl;
      if (i == 0) {
        loopControl = &*outerDoConstruct.GetLoopControl();
        locs.push_back(converter.genLocation(
            Fortran::parser::FindSourceLocation(outerDoConstruct)));
      } else {
        auto *doCons = crtEval->getIf<Fortran::parser::DoConstruct>();
        assert(doCons && "expect do construct");
        loopControl = &*doCons->GetLoopControl();
        locs.push_back(converter.genLocation(
            Fortran::parser::FindSourceLocation(*doCons)));
      }

      const Fortran::parser::LoopControl::Bounds *bounds =
          std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
      assert(bounds && "Expected bounds on the loop construct");
      lowerbounds.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->lower), stmtCtx)));
      upperbounds.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->upper), stmtCtx)));
      if (bounds->step)
        steps.push_back(fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(bounds->step), stmtCtx)));
      else // If `step` is not present, assume it is `1`.
        steps.push_back(builder.createIntegerConstant(
            currentLocation, upperbounds[upperbounds.size() - 1].getType(), 1));

      Fortran::semantics::Symbol &ivSym =
          bounds->name.thing.symbol->GetUltimate();
      privatizeIv(converter, ivSym, currentLocation, ivTypes, ivLocs,
                  privateOperands, ivPrivate, privatizationRecipes);

      inclusiveBounds.push_back(true);

      if (i < collapseValue - 1)
        crtEval = &*std::next(crtEval->getNestedEvaluations().begin());
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperands(operands, operandSegments, lowerbounds);
  addOperands(operands, operandSegments, upperbounds);
  addOperands(operands, operandSegments, steps);
  addOperands(operands, operandSegments, gangOperands);
  addOperands(operands, operandSegments, workerNumOperands);
  addOperands(operands, operandSegments, vectorOperands);
  addOperands(operands, operandSegments, tileOperands);
  addOperands(operands, operandSegments, cacheOperands);
  addOperands(operands, operandSegments, privateOperands);
  addOperands(operands, operandSegments, reductionOperands);

  llvm::SmallVector<mlir::Type> retTy;
  mlir::Value yieldValue;
  if (needEarlyReturnHandling) {
    mlir::Type i1Ty = builder.getI1Type();
    yieldValue = builder.createIntegerConstant(currentLocation, i1Ty, 0);
    retTy.push_back(i1Ty);
  }

  auto loopOp = createRegionOp<mlir::acc::LoopOp, mlir::acc::YieldOp>(
      builder, builder.getFusedLoc(locs), currentLocation, eval, operands,
      operandSegments, /*outerCombined=*/false, retTy, yieldValue, ivTypes,
      ivLocs);

  for (auto [arg, value] : llvm::zip(
           loopOp.getLoopRegions().front()->front().getArguments(), ivPrivate))
    builder.create<fir::StoreOp>(currentLocation, arg, value);

  loopOp.setInclusiveUpperbound(inclusiveBounds);

  if (!gangDeviceTypes.empty())
    loopOp.setGangAttr(builder.getArrayAttr(gangDeviceTypes));
  if (!gangArgTypes.empty())
    loopOp.setGangOperandsArgTypeAttr(builder.getArrayAttr(gangArgTypes));
  if (!gangOperandsSegments.empty())
    loopOp.setGangOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(gangOperandsSegments));
  if (!gangOperandsDeviceTypes.empty())
    loopOp.setGangOperandsDeviceTypeAttr(
        builder.getArrayAttr(gangOperandsDeviceTypes));

  if (!workerNumDeviceTypes.empty())
    loopOp.setWorkerAttr(builder.getArrayAttr(workerNumDeviceTypes));
  if (!workerNumOperandsDeviceTypes.empty())
    loopOp.setWorkerNumOperandsDeviceTypeAttr(
        builder.getArrayAttr(workerNumOperandsDeviceTypes));

  if (!vectorDeviceTypes.empty())
    loopOp.setVectorAttr(builder.getArrayAttr(vectorDeviceTypes));
  if (!vectorOperandsDeviceTypes.empty())
    loopOp.setVectorOperandsDeviceTypeAttr(
        builder.getArrayAttr(vectorOperandsDeviceTypes));

  if (!tileOperandsDeviceTypes.empty())
    loopOp.setTileOperandsDeviceTypeAttr(
        builder.getArrayAttr(tileOperandsDeviceTypes));
  if (!tileOperandsSegments.empty())
    loopOp.setTileOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(tileOperandsSegments));

  // Determine the loop's default par mode - either seq, independent, or auto.
  determineDefaultLoopParMode(converter, loopOp, seqDeviceTypes,
                              independentDeviceTypes, autoDeviceTypes);
  if (!seqDeviceTypes.empty())
    loopOp.setSeqAttr(builder.getArrayAttr(seqDeviceTypes));
  if (!independentDeviceTypes.empty())
    loopOp.setIndependentAttr(builder.getArrayAttr(independentDeviceTypes));
  if (!autoDeviceTypes.empty())
    loopOp.setAuto_Attr(builder.getArrayAttr(autoDeviceTypes));

  if (!privatizationRecipes.empty())
    loopOp.setPrivatizationRecipesAttr(
        mlir::ArrayAttr::get(builder.getContext(), privatizationRecipes));

  if (!reductionRecipes.empty())
    loopOp.setReductionRecipesAttr(
        mlir::ArrayAttr::get(builder.getContext(), reductionRecipes));

  if (!collapseValues.empty())
    loopOp.setCollapseAttr(builder.getI64ArrayAttr(collapseValues));
  if (!collapseDeviceTypes.empty())
    loopOp.setCollapseDeviceTypeAttr(builder.getArrayAttr(collapseDeviceTypes));

  if (combinedConstructs)
    loopOp.setCombinedAttr(mlir::acc::CombinedConstructsTypeAttr::get(
        builder.getContext(), *combinedConstructs));

  // TODO: retrieve directives from NonLabelDoStmt pft::Evaluation, and add them
  // as attribute to the acc.loop as an extra attribute. It is not quite clear
  // how useful these $dir are in acc contexts, but they could still provide
  // more information about the loop acc codegen. They can be obtained by
  // looking for the first lexicalSuccessor of eval that is a NonLabelDoStmt,
  // and using the related `dirs` member.

  return loopOp;
}

static bool hasEarlyReturn(Fortran::lower::pft::Evaluation &eval) {
  bool hasReturnStmt = false;
  for (auto &e : eval.getNestedEvaluations()) {
    e.visit(Fortran::common::visitors{
        [&](const Fortran::parser::ReturnStmt &) { hasReturnStmt = true; },
        [&](const auto &s) {},
    });
    if (e.hasNestedEvaluations())
      hasReturnStmt = hasEarlyReturn(e);
  }
  return hasReturnStmt;
}

static mlir::Value
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {

  const auto &beginLoopDirective =
      std::get<Fortran::parser::AccBeginLoopDirective>(loopConstruct.t);
  const auto &loopDirective =
      std::get<Fortran::parser::AccLoopDirective>(beginLoopDirective.t);

  bool needEarlyExitHandling = false;
  if (eval.lowerAsUnstructured())
    needEarlyExitHandling = hasEarlyReturn(eval);

  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  assert(loopDirective.v == llvm::acc::ACCD_loop &&
         "Unsupported OpenACC loop construct");
  (void)loopDirective;

  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(beginLoopDirective.t);
  const auto &outerDoConstruct =
      std::get<std::optional<Fortran::parser::DoConstruct>>(loopConstruct.t);
  auto loopOp = createLoopOp(converter, currentLocation, semanticsContext,
                             stmtCtx, *outerDoConstruct, eval, accClauseList,
                             /*combinedConstructs=*/{}, needEarlyExitHandling);
  if (needEarlyExitHandling)
    return loopOp.getResult(0);

  return mlir::Value{};
}

template <typename Op, typename Clause>
static void genDataOperandOperationsWithModifier(
    const Clause *x, Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::parser::AccDataModifier::Modifier mod,
    llvm::SmallVectorImpl<mlir::Value> &dataClauseOperands,
    const mlir::acc::DataClause clause,
    const mlir::acc::DataClause clauseWithModifier,
    llvm::ArrayRef<mlir::Value> async,
    llvm::ArrayRef<mlir::Attribute> asyncDeviceTypes,
    llvm::ArrayRef<mlir::Attribute> asyncOnlyDeviceTypes,
    bool setDeclareAttr = false) {
  const Fortran::parser::AccObjectListWithModifier &listWithModifier = x->v;
  const auto &accObjectList =
      std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
  const auto &modifier =
      std::get<std::optional<Fortran::parser::AccDataModifier>>(
          listWithModifier.t);
  mlir::acc::DataClause dataClause =
      (modifier && (*modifier).v == mod) ? clauseWithModifier : clause;
  genDataOperandOperations<Op>(accObjectList, converter, semanticsContext,
                               stmtCtx, dataClauseOperands, dataClause,
                               /*structured=*/true, /*implicit=*/false, async,
                               asyncDeviceTypes, asyncOnlyDeviceTypes,
                               setDeclareAttr);
}

template <typename Op>
static Op createComputeOp(
    Fortran::lower::AbstractConverter &converter,
    mlir::Location currentLocation, Fortran::lower::pft::Evaluation &eval,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::AccClauseList &accClauseList,
    std::optional<mlir::acc::CombinedConstructsType> combinedConstructs =
        std::nullopt) {

  // Parallel operation operands
  mlir::Value ifCond;
  mlir::Value selfCond;
  llvm::SmallVector<mlir::Value> waitOperands, attachEntryOperands,
      copyEntryOperands, copyinEntryOperands, copyoutEntryOperands,
      createEntryOperands, nocreateEntryOperands, presentEntryOperands,
      dataClauseOperands, numGangs, numWorkers, vectorLength, async;
  llvm::SmallVector<mlir::Attribute> numGangsDeviceTypes, numWorkersDeviceTypes,
      vectorLengthDeviceTypes, asyncDeviceTypes, asyncOnlyDeviceTypes,
      waitOperandsDeviceTypes, waitOnlyDeviceTypes;
  llvm::SmallVector<int32_t> numGangsSegments, waitOperandsSegments;
  llvm::SmallVector<bool> hasWaitDevnums;

  llvm::SmallVector<mlir::Value> reductionOperands, privateOperands,
      firstprivateOperands;
  llvm::SmallVector<mlir::Attribute> privatizationRecipes,
      firstPrivatizationRecipes, reductionRecipes;

  // Self clause has optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addSelfAttr = false;

  bool hasDefaultNone = false;
  bool hasDefaultPresent = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  auto crtDeviceTypeAttr = mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None);
  crtDeviceTypes.push_back(crtDeviceTypeAttr);

  // Lower clauses values mapped to operands and array attributes.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.

  // Process the clauses that may have a specified device_type first.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, asyncDeviceTypes,
                     asyncOnlyDeviceTypes, crtDeviceTypes, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClauseWithDeviceType(converter, waitClause, waitOperands,
                                  waitOperandsDeviceTypes, waitOnlyDeviceTypes,
                                  hasWaitDevnums, waitOperandsSegments,
                                  crtDeviceTypes, stmtCtx);
    } else if (const auto *numGangsClause =
                   std::get_if<Fortran::parser::AccClause::NumGangs>(
                       &clause.u)) {
      llvm::SmallVector<mlir::Value> numGangValues;
      for (const Fortran::parser::ScalarIntExpr &expr : numGangsClause->v)
        numGangValues.push_back(fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(expr), stmtCtx)));
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        for (auto value : numGangValues)
          numGangs.push_back(value);
        numGangsDeviceTypes.push_back(crtDeviceTypeAttr);
        numGangsSegments.push_back(numGangValues.size());
      }
    } else if (const auto *numWorkersClause =
                   std::get_if<Fortran::parser::AccClause::NumWorkers>(
                       &clause.u)) {
      mlir::Value numWorkerValue = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numWorkersClause->v), stmtCtx));
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        numWorkers.push_back(numWorkerValue);
        numWorkersDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *vectorLengthClause =
                   std::get_if<Fortran::parser::AccClause::VectorLength>(
                       &clause.u)) {
      mlir::Value vectorLengthValue = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(vectorLengthClause->v), stmtCtx));
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        vectorLength.push_back(vectorLengthValue);
        vectorLengthDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    }
  }

  // Process the clauses independent of device_type.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *selfClause =
                   std::get_if<Fortran::parser::AccClause::Self>(&clause.u)) {
      const std::optional<Fortran::parser::AccSelfClause> &accSelfClause =
          selfClause->v;
      if (accSelfClause) {
        if (const auto *optCondition =
                std::get_if<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                    &(*accSelfClause).u)) {
          if (*optCondition) {
            mlir::Value cond = fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(*optCondition), stmtCtx));
            selfCond = builder.createConvert(clauseLocation,
                                             builder.getI1Type(), cond);
          }
        } else if (const auto *accClauseList =
                       std::get_if<Fortran::parser::AccObjectList>(
                           &(*accSelfClause).u)) {
          // TODO This would be nicer to be done in canonicalization step.
          if (accClauseList->v.size() == 1) {
            const auto &accObject = accClauseList->v.front();
            if (const auto *designator =
                    std::get_if<Fortran::parser::Designator>(&accObject.u)) {
              if (const auto *name =
                      Fortran::semantics::getDesignatorNameIfDataRef(
                          *designator)) {
                auto cond = converter.getSymbolAddress(*name->symbol);
                selfCond = builder.createConvert(clauseLocation,
                                                 builder.getI1Type(), cond);
              }
            }
          }
        }
      } else {
        addSelfAttr = true;
      }
    } else if (const auto *copyClause =
                   std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::CopyinOp>(
          copyClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copy,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                               dataClauseOperands.end());
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CopyinOp,
                                           Fortran::parser::AccClause::Copyin>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      copyinEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyout,
          mlir::acc::DataClause::acc_copyout_zero, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      copyoutEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Create>(
          createClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, dataClauseOperands,
          mlir::acc::DataClause::acc_create,
          mlir::acc::DataClause::acc_create_zero, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      createEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::NoCreateOp>(
          noCreateClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_no_create,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      nocreateEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                   dataClauseOperands.end());
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::PresentOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      presentEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *devicePtrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::DevicePtrOp>(
          devicePtrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      attachEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *privateClause =
                   std::get_if<Fortran::parser::AccClause::Private>(
                       &clause.u)) {
      if (!combinedConstructs)
        genPrivatizationRecipes<mlir::acc::PrivateRecipeOp>(
            privateClause->v, converter, semanticsContext, stmtCtx,
            privateOperands, privatizationRecipes, async, asyncDeviceTypes,
            asyncOnlyDeviceTypes);
    } else if (const auto *firstprivateClause =
                   std::get_if<Fortran::parser::AccClause::Firstprivate>(
                       &clause.u)) {
      genPrivatizationRecipes<mlir::acc::FirstprivateRecipeOp>(
          firstprivateClause->v, converter, semanticsContext, stmtCtx,
          firstprivateOperands, firstPrivatizationRecipes, async,
          asyncDeviceTypes, asyncOnlyDeviceTypes);
    } else if (const auto *reductionClause =
                   std::get_if<Fortran::parser::AccClause::Reduction>(
                       &clause.u)) {
      // A reduction clause on a combined construct is treated as if it appeared
      // on the loop construct. So don't generate a reduction clause when it is
      // combined - delay it to the loop. However, a reduction clause on a
      // combined construct implies a copy clause so issue an implicit copy
      // instead.
      if (!combinedConstructs) {
        genReductions(reductionClause->v, converter, semanticsContext, stmtCtx,
                      reductionOperands, reductionRecipes, async,
                      asyncDeviceTypes, asyncOnlyDeviceTypes);
      } else {
        auto crtDataStart = dataClauseOperands.size();
        genDataOperandOperations<mlir::acc::CopyinOp>(
            std::get<Fortran::parser::AccObjectList>(reductionClause->v.t),
            converter, semanticsContext, stmtCtx, dataClauseOperands,
            mlir::acc::DataClause::acc_reduction,
            /*structured=*/true, /*implicit=*/true, async, asyncDeviceTypes,
            asyncOnlyDeviceTypes);
        copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
      }
    } else if (const auto *defaultClause =
                   std::get_if<Fortran::parser::AccClause::Default>(
                       &clause.u)) {
      if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_none)
        hasDefaultNone = true;
      else if ((defaultClause->v).v ==
               llvm::acc::DefaultValue::ACC_Default_present)
        hasDefaultPresent = true;
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;
  addOperands(operands, operandSegments, async);
  addOperands(operands, operandSegments, waitOperands);
  if constexpr (!std::is_same_v<Op, mlir::acc::SerialOp>) {
    addOperands(operands, operandSegments, numGangs);
    addOperands(operands, operandSegments, numWorkers);
    addOperands(operands, operandSegments, vectorLength);
  }
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, selfCond);
  if constexpr (!std::is_same_v<Op, mlir::acc::KernelsOp>) {
    addOperands(operands, operandSegments, reductionOperands);
    addOperands(operands, operandSegments, privateOperands);
    addOperands(operands, operandSegments, firstprivateOperands);
  }
  addOperands(operands, operandSegments, dataClauseOperands);

  Op computeOp;
  if constexpr (std::is_same_v<Op, mlir::acc::KernelsOp>)
    computeOp = createRegionOp<Op, mlir::acc::TerminatorOp>(
        builder, currentLocation, currentLocation, eval, operands,
        operandSegments, /*outerCombined=*/combinedConstructs.has_value());
  else
    computeOp = createRegionOp<Op, mlir::acc::YieldOp>(
        builder, currentLocation, currentLocation, eval, operands,
        operandSegments, /*outerCombined=*/combinedConstructs.has_value());

  if (addSelfAttr)
    computeOp.setSelfAttrAttr(builder.getUnitAttr());

  if (hasDefaultNone)
    computeOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
  if (hasDefaultPresent)
    computeOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);

  if constexpr (!std::is_same_v<Op, mlir::acc::SerialOp>) {
    if (!numWorkersDeviceTypes.empty())
      computeOp.setNumWorkersDeviceTypeAttr(
          mlir::ArrayAttr::get(builder.getContext(), numWorkersDeviceTypes));
    if (!vectorLengthDeviceTypes.empty())
      computeOp.setVectorLengthDeviceTypeAttr(
          mlir::ArrayAttr::get(builder.getContext(), vectorLengthDeviceTypes));
    if (!numGangsDeviceTypes.empty())
      computeOp.setNumGangsDeviceTypeAttr(
          mlir::ArrayAttr::get(builder.getContext(), numGangsDeviceTypes));
    if (!numGangsSegments.empty())
      computeOp.setNumGangsSegmentsAttr(
          builder.getDenseI32ArrayAttr(numGangsSegments));
  }
  if (!asyncDeviceTypes.empty())
    computeOp.setAsyncOperandsDeviceTypeAttr(
        builder.getArrayAttr(asyncDeviceTypes));
  if (!asyncOnlyDeviceTypes.empty())
    computeOp.setAsyncOnlyAttr(builder.getArrayAttr(asyncOnlyDeviceTypes));

  if (!waitOperandsDeviceTypes.empty())
    computeOp.setWaitOperandsDeviceTypeAttr(
        builder.getArrayAttr(waitOperandsDeviceTypes));
  if (!waitOperandsSegments.empty())
    computeOp.setWaitOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(waitOperandsSegments));
  if (!hasWaitDevnums.empty())
    computeOp.setHasWaitDevnumAttr(builder.getBoolArrayAttr(hasWaitDevnums));
  if (!waitOnlyDeviceTypes.empty())
    computeOp.setWaitOnlyAttr(builder.getArrayAttr(waitOnlyDeviceTypes));

  if constexpr (!std::is_same_v<Op, mlir::acc::KernelsOp>) {
    if (!privatizationRecipes.empty())
      computeOp.setPrivatizationRecipesAttr(
          mlir::ArrayAttr::get(builder.getContext(), privatizationRecipes));
    if (!reductionRecipes.empty())
      computeOp.setReductionRecipesAttr(
          mlir::ArrayAttr::get(builder.getContext(), reductionRecipes));
    if (!firstPrivatizationRecipes.empty())
      computeOp.setFirstprivatizationRecipesAttr(mlir::ArrayAttr::get(
          builder.getContext(), firstPrivatizationRecipes));
  }

  if (combinedConstructs)
    computeOp.setCombinedAttr(builder.getUnitAttr());

  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(computeOp);

  // Create the exit operations after the region.
  genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
      builder, copyEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::DeleteOp>(
      builder, copyinEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
      builder, copyoutEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::AttachOp, mlir::acc::DetachOp>(
      builder, attachEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
      builder, createEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::NoCreateOp, mlir::acc::DeleteOp>(
      builder, nocreateEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::PresentOp, mlir::acc::DeleteOp>(
      builder, presentEntryOperands, /*structured=*/true);

  builder.restoreInsertionPoint(insPt);
  return computeOp;
}

static void genACCDataOp(Fortran::lower::AbstractConverter &converter,
                         mlir::Location currentLocation,
                         mlir::Location endLocation,
                         Fortran::lower::pft::Evaluation &eval,
                         Fortran::semantics::SemanticsContext &semanticsContext,
                         Fortran::lower::StatementContext &stmtCtx,
                         const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond;
  llvm::SmallVector<mlir::Value> attachEntryOperands, createEntryOperands,
      copyEntryOperands, copyinEntryOperands, copyoutEntryOperands,
      nocreateEntryOperands, presentEntryOperands, dataClauseOperands,
      waitOperands, async;
  llvm::SmallVector<mlir::Attribute> asyncDeviceTypes, asyncOnlyDeviceTypes,
      waitOperandsDeviceTypes, waitOnlyDeviceTypes;
  llvm::SmallVector<int32_t> waitOperandsSegments;
  llvm::SmallVector<bool> hasWaitDevnums;

  bool hasDefaultNone = false;
  bool hasDefaultPresent = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  crtDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None));

  // Lower clauses values mapped to operands and array attributes.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.

  // Process the clauses that may have a specified device_type first.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, asyncDeviceTypes,
                     asyncOnlyDeviceTypes, crtDeviceTypes, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClauseWithDeviceType(converter, waitClause, waitOperands,
                                  waitOperandsDeviceTypes, waitOnlyDeviceTypes,
                                  hasWaitDevnums, waitOperandsSegments,
                                  crtDeviceTypes, stmtCtx);
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    }
  }

  // Process the clauses independent of device_type.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *copyClause =
                   std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::CopyinOp>(
          copyClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copy,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                               dataClauseOperands.end());
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CopyinOp,
                                           Fortran::parser::AccClause::Copyin>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      copyinEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, dataClauseOperands,
          mlir::acc::DataClause::acc_copyout,
          mlir::acc::DataClause::acc_copyout_zero, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      copyoutEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Create>(
          createClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, dataClauseOperands,
          mlir::acc::DataClause::acc_create,
          mlir::acc::DataClause::acc_create_zero, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      createEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::NoCreateOp>(
          noCreateClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_no_create,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      nocreateEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                   dataClauseOperands.end());
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::PresentOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      presentEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *deviceptrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::DevicePtrOp>(
          deviceptrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach,
          /*structured=*/true, /*implicit=*/false, async, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
      attachEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *defaultClause =
                   std::get_if<Fortran::parser::AccClause::Default>(
                       &clause.u)) {
      if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_none)
        hasDefaultNone = true;
      else if ((defaultClause->v).v ==
               llvm::acc::DefaultValue::ACC_Default_present)
        hasDefaultPresent = true;
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, async);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  if (dataClauseOperands.empty() && !hasDefaultNone && !hasDefaultPresent)
    return;

  auto dataOp = createRegionOp<mlir::acc::DataOp, mlir::acc::TerminatorOp>(
      builder, currentLocation, currentLocation, eval, operands,
      operandSegments);

  if (!asyncDeviceTypes.empty())
    dataOp.setAsyncOperandsDeviceTypeAttr(
        builder.getArrayAttr(asyncDeviceTypes));
  if (!asyncOnlyDeviceTypes.empty())
    dataOp.setAsyncOnlyAttr(builder.getArrayAttr(asyncOnlyDeviceTypes));
  if (!waitOperandsDeviceTypes.empty())
    dataOp.setWaitOperandsDeviceTypeAttr(
        builder.getArrayAttr(waitOperandsDeviceTypes));
  if (!waitOperandsSegments.empty())
    dataOp.setWaitOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(waitOperandsSegments));
  if (!hasWaitDevnums.empty())
    dataOp.setHasWaitDevnumAttr(builder.getBoolArrayAttr(hasWaitDevnums));
  if (!waitOnlyDeviceTypes.empty())
    dataOp.setWaitOnlyAttr(builder.getArrayAttr(waitOnlyDeviceTypes));

  if (hasDefaultNone)
    dataOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
  if (hasDefaultPresent)
    dataOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);

  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(dataOp);

  // Create the exit operations after the region.
  genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
      builder, copyEntryOperands, /*structured=*/true, endLocation);
  genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::DeleteOp>(
      builder, copyinEntryOperands, /*structured=*/true, endLocation);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
      builder, copyoutEntryOperands, /*structured=*/true, endLocation);
  genDataExitOperations<mlir::acc::AttachOp, mlir::acc::DetachOp>(
      builder, attachEntryOperands, /*structured=*/true, endLocation);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
      builder, createEntryOperands, /*structured=*/true, endLocation);
  genDataExitOperations<mlir::acc::NoCreateOp, mlir::acc::DeleteOp>(
      builder, nocreateEntryOperands, /*structured=*/true, endLocation);
  genDataExitOperations<mlir::acc::PresentOp, mlir::acc::DeleteOp>(
      builder, presentEntryOperands, /*structured=*/true, endLocation);

  builder.restoreInsertionPoint(insPt);
}

static void
genACCHostDataOp(Fortran::lower::AbstractConverter &converter,
                 mlir::Location currentLocation,
                 Fortran::lower::pft::Evaluation &eval,
                 Fortran::semantics::SemanticsContext &semanticsContext,
                 Fortran::lower::StatementContext &stmtCtx,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond;
  llvm::SmallVector<mlir::Value> dataOperands;
  bool addIfPresentAttr = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *useDevice =
                   std::get_if<Fortran::parser::AccClause::UseDevice>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::UseDeviceOp>(
          useDevice->v, converter, semanticsContext, stmtCtx, dataOperands,
          mlir::acc::DataClause::acc_use_device,
          /*structured=*/true, /*implicit=*/false, /*async=*/{},
          /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
    } else if (std::get_if<Fortran::parser::AccClause::IfPresent>(&clause.u)) {
      addIfPresentAttr = true;
    }
  }

  if (ifCond) {
    if (auto cst =
            mlir::dyn_cast<mlir::arith::ConstantOp>(ifCond.getDefiningOp()))
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(cst.getValue())) {
        if (boolAttr.getValue()) {
          // get rid of the if condition if it is always true.
          ifCond = mlir::Value();
        } else {
          // Do not generate the acc.host_data op if the if condition is always
          // false.
          return;
        }
      }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, dataOperands);

  auto hostDataOp =
      createRegionOp<mlir::acc::HostDataOp, mlir::acc::TerminatorOp>(
          builder, currentLocation, currentLocation, eval, operands,
          operandSegments);

  if (addIfPresentAttr)
    hostDataOp.setIfPresentAttr(builder.getUnitAttr());
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::AccBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::AccBlockDirective>(beginBlockDirective.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(beginBlockDirective.t);
  const auto &endBlockDirective =
      std::get<Fortran::parser::AccEndBlockDirective>(blockConstruct.t);
  mlir::Location endLocation = converter.genLocation(endBlockDirective.source);
  mlir::Location currentLocation = converter.genLocation(blockDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (blockDirective.v == llvm::acc::ACCD_parallel) {
    createComputeOp<mlir::acc::ParallelOp>(converter, currentLocation, eval,
                                           semanticsContext, stmtCtx,
                                           accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_data) {
    genACCDataOp(converter, currentLocation, endLocation, eval,
                 semanticsContext, stmtCtx, accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_serial) {
    createComputeOp<mlir::acc::SerialOp>(converter, currentLocation, eval,
                                         semanticsContext, stmtCtx,
                                         accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_kernels) {
    createComputeOp<mlir::acc::KernelsOp>(converter, currentLocation, eval,
                                          semanticsContext, stmtCtx,
                                          accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_host_data) {
    genACCHostDataOp(converter, currentLocation, eval, semanticsContext,
                     stmtCtx, accClauseList);
  }
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCCombinedConstruct &combinedConstruct) {
  const auto &beginCombinedDirective =
      std::get<Fortran::parser::AccBeginCombinedDirective>(combinedConstruct.t);
  const auto &combinedDirective =
      std::get<Fortran::parser::AccCombinedDirective>(beginCombinedDirective.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(beginCombinedDirective.t);
  const auto &outerDoConstruct =
      std::get<std::optional<Fortran::parser::DoConstruct>>(
          combinedConstruct.t);

  mlir::Location currentLocation =
      converter.genLocation(beginCombinedDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (combinedDirective.v == llvm::acc::ACCD_kernels_loop) {
    createComputeOp<mlir::acc::KernelsOp>(
        converter, currentLocation, eval, semanticsContext, stmtCtx,
        accClauseList, mlir::acc::CombinedConstructsType::KernelsLoop);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 *outerDoConstruct, eval, accClauseList,
                 mlir::acc::CombinedConstructsType::KernelsLoop);
  } else if (combinedDirective.v == llvm::acc::ACCD_parallel_loop) {
    createComputeOp<mlir::acc::ParallelOp>(
        converter, currentLocation, eval, semanticsContext, stmtCtx,
        accClauseList, mlir::acc::CombinedConstructsType::ParallelLoop);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 *outerDoConstruct, eval, accClauseList,
                 mlir::acc::CombinedConstructsType::ParallelLoop);
  } else if (combinedDirective.v == llvm::acc::ACCD_serial_loop) {
    createComputeOp<mlir::acc::SerialOp>(
        converter, currentLocation, eval, semanticsContext, stmtCtx,
        accClauseList, mlir::acc::CombinedConstructsType::SerialLoop);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 *outerDoConstruct, eval, accClauseList,
                 mlir::acc::CombinedConstructsType::SerialLoop);
  } else {
    llvm::report_fatal_error("Unknown combined construct encountered");
  }
}

static void
genACCEnterDataOp(Fortran::lower::AbstractConverter &converter,
                  mlir::Location currentLocation,
                  Fortran::semantics::SemanticsContext &semanticsContext,
                  Fortran::lower::StatementContext &stmtCtx,
                  const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  llvm::SmallVector<mlir::Value> waitOperands, dataClauseOperands;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.

  // Process the async clause first.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    }
  }

  // The async clause of 'enter data' applies to all device types,
  // so propagate the async clause to copyin/create/attach ops
  // as if it is an async clause without preceding device_type clause.
  llvm::SmallVector<mlir::Attribute> asyncDeviceTypes, asyncOnlyDeviceTypes;
  llvm::SmallVector<mlir::Value> asyncValues;
  auto noneDeviceTypeAttr = mlir::acc::DeviceTypeAttr::get(
      firOpBuilder.getContext(), mlir::acc::DeviceType::None);
  if (addAsyncAttr) {
    asyncOnlyDeviceTypes.push_back(noneDeviceTypeAttr);
  } else if (async) {
    asyncValues.push_back(async);
    asyncDeviceTypes.push_back(noneDeviceTypeAttr);
  }

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands, waitDevnum,
                    addWaitAttr, stmtCtx);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyinClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genDataOperandOperations<mlir::acc::CopyinOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin, false,
          /*implicit=*/false, asyncValues, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          createClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      const auto &modifier =
          std::get<std::optional<Fortran::parser::AccDataModifier>>(
              listWithModifier.t);
      mlir::acc::DataClause clause = mlir::acc::DataClause::acc_create;
      if (modifier &&
          (*modifier).v == Fortran::parser::AccDataModifier::Modifier::Zero)
        clause = mlir::acc::DataClause::acc_create_zero;
      genDataOperandOperations<mlir::acc::CreateOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, clause, false, /*implicit=*/false, asyncValues,
          asyncDeviceTypes, asyncOnlyDeviceTypes);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach, false,
          /*implicit=*/false, asyncValues, asyncDeviceTypes,
          asyncOnlyDeviceTypes);
    } else if (!std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      llvm::report_fatal_error(
          "Unknown clause in ENTER DATA directive lowering");
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 16> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  mlir::acc::EnterDataOp enterDataOp = createSimpleOp<mlir::acc::EnterDataOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    enterDataOp.setAsyncAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    enterDataOp.setWaitAttr(firOpBuilder.getUnitAttr());
}

static void
genACCExitDataOp(Fortran::lower::AbstractConverter &converter,
                 mlir::Location currentLocation,
                 Fortran::semantics::SemanticsContext &semanticsContext,
                 Fortran::lower::StatementContext &stmtCtx,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  llvm::SmallVector<mlir::Value> waitOperands, dataClauseOperands,
      copyoutOperands, deleteOperands, detachOperands;

  // Async and wait clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addFinalizeAttr = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.

  // Process the async clause first.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    }
  }

  // The async clause of 'exit data' applies to all device types,
  // so propagate the async clause to copyin/create/attach ops
  // as if it is an async clause without preceding device_type clause.
  llvm::SmallVector<mlir::Attribute> asyncDeviceTypes, asyncOnlyDeviceTypes;
  llvm::SmallVector<mlir::Value> asyncValues;
  auto noneDeviceTypeAttr = mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None);
  if (addAsyncAttr) {
    asyncOnlyDeviceTypes.push_back(noneDeviceTypeAttr);
  } else if (async) {
    asyncValues.push_back(async);
    asyncDeviceTypes.push_back(noneDeviceTypeAttr);
  }

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands, waitDevnum,
                    addWaitAttr, stmtCtx);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyoutClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          accObjectList, converter, semanticsContext, stmtCtx, copyoutOperands,
          mlir::acc::DataClause::acc_copyout, false, /*implicit=*/false,
          asyncValues, asyncDeviceTypes, asyncOnlyDeviceTypes);
    } else if (const auto *deleteClause =
                   std::get_if<Fortran::parser::AccClause::Delete>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          deleteClause->v, converter, semanticsContext, stmtCtx, deleteOperands,
          mlir::acc::DataClause::acc_delete, false, /*implicit=*/false,
          asyncValues, asyncDeviceTypes, asyncOnlyDeviceTypes);
    } else if (const auto *detachClause =
                   std::get_if<Fortran::parser::AccClause::Detach>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          detachClause->v, converter, semanticsContext, stmtCtx, detachOperands,
          mlir::acc::DataClause::acc_detach, false, /*implicit=*/false,
          asyncValues, asyncDeviceTypes, asyncOnlyDeviceTypes);
    } else if (std::get_if<Fortran::parser::AccClause::Finalize>(&clause.u)) {
      addFinalizeAttr = true;
    }
  }

  dataClauseOperands.append(copyoutOperands);
  dataClauseOperands.append(deleteOperands);
  dataClauseOperands.append(detachOperands);

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 14> operands;
  llvm::SmallVector<int32_t, 7> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  mlir::acc::ExitDataOp exitDataOp = createSimpleOp<mlir::acc::ExitDataOp>(
      builder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    exitDataOp.setAsyncAttr(builder.getUnitAttr());
  if (addWaitAttr)
    exitDataOp.setWaitAttr(builder.getUnitAttr());
  if (addFinalizeAttr)
    exitDataOp.setFinalizeAttr(builder.getUnitAttr());

  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::CopyoutOp>(
      builder, copyoutOperands, /*structured=*/false);
  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::DeleteOp>(
      builder, deleteOperands, /*structured=*/false);
  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::DetachOp>(
      builder, detachOperands, /*structured=*/false);
}

template <typename Op>
static void
genACCInitShutdownOp(Fortran::lower::AbstractConverter &converter,
                     mlir::Location currentLocation,
                     const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, deviceNum;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<mlir::Attribute> deviceTypes;

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *deviceNumClause =
                   std::get_if<Fortran::parser::AccClause::DeviceNum>(
                       &clause.u)) {
      deviceNum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(deviceNumClause->v), stmtCtx));
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      gatherDeviceTypeAttrs(builder, deviceTypeClause, deviceTypes);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 6> operands;
  llvm::SmallVector<int32_t, 2> operandSegments;

  addOperand(operands, operandSegments, deviceNum);
  addOperand(operands, operandSegments, ifCond);

  Op op =
      createSimpleOp<Op>(builder, currentLocation, operands, operandSegments);
  if (!deviceTypes.empty())
    op.setDeviceTypesAttr(
        mlir::ArrayAttr::get(builder.getContext(), deviceTypes));
}

void genACCSetOp(Fortran::lower::AbstractConverter &converter,
                 mlir::Location currentLocation,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, deviceNum, defaultAsync;
  llvm::SmallVector<mlir::Value> deviceTypeOperands;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<mlir::Attribute> deviceTypes;

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *defaultAsyncClause =
                   std::get_if<Fortran::parser::AccClause::DefaultAsync>(
                       &clause.u)) {
      defaultAsync = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(defaultAsyncClause->v), stmtCtx));
    } else if (const auto *deviceNumClause =
                   std::get_if<Fortran::parser::AccClause::DeviceNum>(
                       &clause.u)) {
      deviceNum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(deviceNumClause->v), stmtCtx));
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      gatherDeviceTypeAttrs(builder, deviceTypeClause, deviceTypes);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t, 3> operandSegments;
  addOperand(operands, operandSegments, defaultAsync);
  addOperand(operands, operandSegments, deviceNum);
  addOperand(operands, operandSegments, ifCond);

  auto op = createSimpleOp<mlir::acc::SetOp>(builder, currentLocation, operands,
                                             operandSegments);
  if (!deviceTypes.empty()) {
    assert(deviceTypes.size() == 1 && "expect only one value for acc.set");
    op.setDeviceTypeAttr(mlir::cast<mlir::acc::DeviceTypeAttr>(deviceTypes[0]));
  }
}

static inline mlir::ArrayAttr
getArrayAttr(fir::FirOpBuilder &b,
             llvm::SmallVector<mlir::Attribute> &attributes) {
  return attributes.empty() ? nullptr : b.getArrayAttr(attributes);
}

static inline mlir::ArrayAttr
getBoolArrayAttr(fir::FirOpBuilder &b, llvm::SmallVector<bool> &values) {
  return values.empty() ? nullptr : b.getBoolArrayAttr(values);
}

static inline mlir::DenseI32ArrayAttr
getDenseI32ArrayAttr(fir::FirOpBuilder &builder,
                     llvm::SmallVector<int32_t> &values) {
  return values.empty() ? nullptr : builder.getDenseI32ArrayAttr(values);
}

static void
genACCUpdateOp(Fortran::lower::AbstractConverter &converter,
               mlir::Location currentLocation,
               Fortran::semantics::SemanticsContext &semanticsContext,
               Fortran::lower::StatementContext &stmtCtx,
               const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond;
  llvm::SmallVector<mlir::Value> dataClauseOperands, updateHostOperands,
      waitOperands, deviceTypeOperands, asyncOperands;
  llvm::SmallVector<mlir::Attribute> asyncOperandsDeviceTypes,
      asyncOnlyDeviceTypes, waitOperandsDeviceTypes, waitOnlyDeviceTypes;
  llvm::SmallVector<bool> hasWaitDevnums;
  llvm::SmallVector<int32_t> waitOperandsSegments;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  crtDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None));

  bool ifPresent = false;

  // Lower clauses values mapped to operands and array attributes.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.

  // Process the clauses that may have a specified device_type first.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, asyncOperands,
                     asyncOperandsDeviceTypes, asyncOnlyDeviceTypes,
                     crtDeviceTypes, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClauseWithDeviceType(converter, waitClause, waitOperands,
                                  waitOperandsDeviceTypes, waitOnlyDeviceTypes,
                                  hasWaitDevnums, waitOperandsSegments,
                                  crtDeviceTypes, stmtCtx);
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    }
  }

  // Process the clauses independent of device_type.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *hostClause =
                   std::get_if<Fortran::parser::AccClause::Host>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          hostClause->v, converter, semanticsContext, stmtCtx,
          updateHostOperands, mlir::acc::DataClause::acc_update_host, false,
          /*implicit=*/false, asyncOperands, asyncOperandsDeviceTypes,
          asyncOnlyDeviceTypes);
    } else if (const auto *deviceClause =
                   std::get_if<Fortran::parser::AccClause::Device>(&clause.u)) {
      genDataOperandOperations<mlir::acc::UpdateDeviceOp>(
          deviceClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_update_device, false,
          /*implicit=*/false, asyncOperands, asyncOperandsDeviceTypes,
          asyncOnlyDeviceTypes);
    } else if (std::get_if<Fortran::parser::AccClause::IfPresent>(&clause.u)) {
      ifPresent = true;
    } else if (const auto *selfClause =
                   std::get_if<Fortran::parser::AccClause::Self>(&clause.u)) {
      const std::optional<Fortran::parser::AccSelfClause> &accSelfClause =
          selfClause->v;
      const auto *accObjectList =
          std::get_if<Fortran::parser::AccObjectList>(&(*accSelfClause).u);
      assert(accObjectList && "expect AccObjectList");
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          *accObjectList, converter, semanticsContext, stmtCtx,
          updateHostOperands, mlir::acc::DataClause::acc_update_self, false,
          /*implicit=*/false, asyncOperands, asyncOperandsDeviceTypes,
          asyncOnlyDeviceTypes);
    }
  }

  dataClauseOperands.append(updateHostOperands);

  builder.create<mlir::acc::UpdateOp>(
      currentLocation, ifCond, asyncOperands,
      getArrayAttr(builder, asyncOperandsDeviceTypes),
      getArrayAttr(builder, asyncOnlyDeviceTypes), waitOperands,
      getDenseI32ArrayAttr(builder, waitOperandsSegments),
      getArrayAttr(builder, waitOperandsDeviceTypes),
      getBoolArrayAttr(builder, hasWaitDevnums),
      getArrayAttr(builder, waitOnlyDeviceTypes), dataClauseOperands,
      ifPresent);

  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::UpdateHostOp>(
      builder, updateHostOperands, /*structured=*/false);
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       const Fortran::parser::OpenACCStandaloneConstruct &standaloneConstruct) {
  const auto &standaloneDirective =
      std::get<Fortran::parser::AccStandaloneDirective>(standaloneConstruct.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(standaloneConstruct.t);

  mlir::Location currentLocation =
      converter.genLocation(standaloneDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (standaloneDirective.v == llvm::acc::Directive::ACCD_enter_data) {
    genACCEnterDataOp(converter, currentLocation, semanticsContext, stmtCtx,
                      accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_exit_data) {
    genACCExitDataOp(converter, currentLocation, semanticsContext, stmtCtx,
                     accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_init) {
    genACCInitShutdownOp<mlir::acc::InitOp>(converter, currentLocation,
                                            accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_shutdown) {
    genACCInitShutdownOp<mlir::acc::ShutdownOp>(converter, currentLocation,
                                                accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_set) {
    genACCSetOp(converter, currentLocation, accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_update) {
    genACCUpdateOp(converter, currentLocation, semanticsContext, stmtCtx,
                   accClauseList);
  }
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {

  const auto &waitArgument =
      std::get<std::optional<Fortran::parser::AccWaitArgument>>(
          waitConstruct.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(waitConstruct.t);

  mlir::Value ifCond, waitDevnum, async;
  llvm::SmallVector<mlir::Value> waitOperands;

  // Async clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.genLocation(waitConstruct.source);
  Fortran::lower::StatementContext stmtCtx;

  if (waitArgument) { // wait has a value.
    const Fortran::parser::AccWaitArgument &waitArg = *waitArgument;
    const auto &waitList =
        std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    for (const Fortran::parser::ScalarIntExpr &value : waitList) {
      mlir::Value v = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(value), stmtCtx));
      waitOperands.push_back(v);
    }

    const auto &waitDevnumValue =
        std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    if (waitDevnumValue)
      waitDevnum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(*waitDevnumValue), stmtCtx));
  }

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperands(operands, operandSegments, waitOperands);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperand(operands, operandSegments, ifCond);

  mlir::acc::WaitOp waitOp = createSimpleOp<mlir::acc::WaitOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    waitOp.setAsyncAttr(firOpBuilder.getUnitAttr());
}

template <typename GlobalOp, typename EntryOp, typename DeclareOp,
          typename ExitOp>
static void createDeclareGlobalOp(mlir::OpBuilder &modBuilder,
                                  fir::FirOpBuilder &builder,
                                  mlir::Location loc, fir::GlobalOp globalOp,
                                  mlir::acc::DataClause clause,
                                  const std::string &declareGlobalName,
                                  bool implicit, std::stringstream &asFortran) {
  GlobalOp declareGlobalOp =
      modBuilder.create<GlobalOp>(loc, declareGlobalName);
  builder.createBlock(&declareGlobalOp.getRegion(),
                      declareGlobalOp.getRegion().end(), {}, {});
  builder.setInsertionPointToEnd(&declareGlobalOp.getRegion().back());

  fir::AddrOfOp addrOp = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());
  addDeclareAttr(builder, addrOp, clause);

  llvm::SmallVector<mlir::Value> bounds;
  EntryOp entryOp = createDataEntryOp<EntryOp>(
      builder, loc, addrOp.getResTy(), asFortran, bounds,
      /*structured=*/false, implicit, clause, addrOp.getResTy().getType(),
      /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
  if constexpr (std::is_same_v<DeclareOp, mlir::acc::DeclareEnterOp>)
    builder.create<DeclareOp>(
        loc, mlir::acc::DeclareTokenType::get(entryOp.getContext()),
        mlir::ValueRange(entryOp.getAccVar()));
  else
    builder.create<DeclareOp>(loc, mlir::Value{},
                              mlir::ValueRange(entryOp.getAccVar()));
  if constexpr (std::is_same_v<GlobalOp, mlir::acc::GlobalDestructorOp>) {
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccVar(),
                           entryOp.getBounds(), entryOp.getAsyncOperands(),
                           entryOp.getAsyncOperandsDeviceTypeAttr(),
                           entryOp.getAsyncOnlyAttr(), entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));
  }
  builder.create<mlir::acc::TerminatorOp>(loc);
  modBuilder.setInsertionPointAfter(declareGlobalOp);
}

template <typename EntryOp>
static void createDeclareAllocFunc(mlir::OpBuilder &modBuilder,
                                   fir::FirOpBuilder &builder,
                                   mlir::Location loc, fir::GlobalOp &globalOp,
                                   mlir::acc::DataClause clause) {
  std::stringstream registerFuncName;
  registerFuncName << globalOp.getSymName().str()
                   << Fortran::lower::declarePostAllocSuffix.str();
  auto registerFuncOp =
      createDeclareFunc(modBuilder, builder, loc, registerFuncName.str());

  fir::AddrOfOp addrOp = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());

  std::stringstream asFortran;
  asFortran << Fortran::lower::mangle::demangleName(globalOp.getSymName());
  std::stringstream asFortranDesc;
  asFortranDesc << asFortran.str();
  if (unwrapFirBox)
    asFortranDesc << accFirDescriptorPostfix.str();
  llvm::SmallVector<mlir::Value> bounds;

  // Updating descriptor must occur before the mapping of the data so that
  // attached data pointer is not overwritten.
  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, addrOp, asFortranDesc, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, addrOp.getType(),
          /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);

  if (unwrapFirBox) {
    auto loadOp = builder.create<fir::LoadOp>(loc, addrOp.getResult());
    fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, loadOp);
    addDeclareAttr(builder, boxAddrOp.getOperation(), clause);
    EntryOp entryOp = createDataEntryOp<EntryOp>(
        builder, loc, boxAddrOp.getResult(), asFortran, bounds,
        /*structured=*/false, /*implicit=*/false, clause, boxAddrOp.getType(),
        /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
    builder.create<mlir::acc::DeclareEnterOp>(
        loc, mlir::acc::DeclareTokenType::get(entryOp.getContext()),
        mlir::ValueRange(entryOp.getAccVar()));
  }

  modBuilder.setInsertionPointAfter(registerFuncOp);
}

/// Action to be performed on deallocation are split in two distinct functions.
/// - Pre deallocation function includes all the action to be performed before
///   the actual deallocation is done on the host side.
/// - Post deallocation function includes update to the descriptor.
template <typename ExitOp>
static void createDeclareDeallocFunc(mlir::OpBuilder &modBuilder,
                                     fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     fir::GlobalOp &globalOp,
                                     mlir::acc::DataClause clause) {
  std::stringstream asFortran;
  asFortran << Fortran::lower::mangle::demangleName(globalOp.getSymName());

  // If FIR box semantics are being unwrapped, then a pre-dealloc function
  // needs generated to ensure to delete the device data pointed to by the
  // descriptor before this information is lost.
  if (unwrapFirBox) {
    // Generate the pre dealloc function.
    std::stringstream preDeallocFuncName;
    preDeallocFuncName << globalOp.getSymName().str()
                       << Fortran::lower::declarePreDeallocSuffix.str();
    auto preDeallocOp =
        createDeclareFunc(modBuilder, builder, loc, preDeallocFuncName.str());

    fir::AddrOfOp addrOp = builder.create<fir::AddrOfOp>(
        loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());
    auto loadOp = builder.create<fir::LoadOp>(loc, addrOp.getResult());
    fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, loadOp);
    mlir::Value var = boxAddrOp.getResult();
    addDeclareAttr(builder, var.getDefiningOp(), clause);

    llvm::SmallVector<mlir::Value> bounds;
    mlir::acc::GetDevicePtrOp entryOp =
        createDataEntryOp<mlir::acc::GetDevicePtrOp>(
            builder, loc, var, asFortran, bounds,
            /*structured=*/false, /*implicit=*/false, clause, var.getType(),
            /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});

    builder.create<mlir::acc::DeclareExitOp>(
        loc, mlir::Value{}, mlir::ValueRange(entryOp.getAccVar()));

    if constexpr (std::is_same_v<ExitOp, mlir::acc::CopyoutOp> ||
                  std::is_same_v<ExitOp, mlir::acc::UpdateHostOp>)
      builder.create<ExitOp>(
          entryOp.getLoc(), entryOp.getAccVar(), entryOp.getVar(),
          entryOp.getBounds(), entryOp.getAsyncOperands(),
          entryOp.getAsyncOperandsDeviceTypeAttr(), entryOp.getAsyncOnlyAttr(),
          entryOp.getDataClause(),
          /*structured=*/false, /*implicit=*/false,
          builder.getStringAttr(*entryOp.getName()));
    else
      builder.create<ExitOp>(
          entryOp.getLoc(), entryOp.getAccVar(), entryOp.getBounds(),
          entryOp.getAsyncOperands(), entryOp.getAsyncOperandsDeviceTypeAttr(),
          entryOp.getAsyncOnlyAttr(), entryOp.getDataClause(),
          /*structured=*/false, /*implicit=*/false,
          builder.getStringAttr(*entryOp.getName()));

    // Generate the post dealloc function.
    modBuilder.setInsertionPointAfter(preDeallocOp);
  }

  std::stringstream postDeallocFuncName;
  postDeallocFuncName << globalOp.getSymName().str()
                      << Fortran::lower::declarePostDeallocSuffix.str();
  auto postDeallocOp =
      createDeclareFunc(modBuilder, builder, loc, postDeallocFuncName.str());

  fir::AddrOfOp addrOp = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());
  if (unwrapFirBox)
    asFortran << accFirDescriptorPostfix.str();
  llvm::SmallVector<mlir::Value> bounds;
  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, addrOp, asFortran, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, addrOp.getType(),
          /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{});
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);
  modBuilder.setInsertionPointAfter(postDeallocOp);
}

template <typename EntryOp, typename ExitOp>
static void genGlobalCtors(Fortran::lower::AbstractConverter &converter,
                           mlir::OpBuilder &modBuilder,
                           const Fortran::parser::AccObjectList &accObjectList,
                           mlir::acc::DataClause clause) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto genCtors = [&](const mlir::Location operandLocation,
                      const Fortran::semantics::Symbol &symbol) {
    std::string globalName = converter.mangleName(symbol);
    fir::GlobalOp globalOp = builder.getNamedGlobal(globalName);
    std::stringstream declareGlobalCtorName;
    declareGlobalCtorName << globalName << "_acc_ctor";
    std::stringstream declareGlobalDtorName;
    declareGlobalDtorName << globalName << "_acc_dtor";
    std::stringstream asFortran;
    asFortran << symbol.name().ToString();

    if (builder.getModule().lookupSymbol<mlir::acc::GlobalConstructorOp>(
            declareGlobalCtorName.str()))
      return;

    if (!globalOp) {
      if (Fortran::semantics::FindEquivalenceSet(symbol)) {
        for (Fortran::semantics::EquivalenceObject eqObj :
             *Fortran::semantics::FindEquivalenceSet(symbol)) {
          std::string eqName = converter.mangleName(eqObj.symbol);
          globalOp = builder.getNamedGlobal(eqName);
          if (globalOp)
            break;
        }

        if (!globalOp)
          llvm::report_fatal_error("could not retrieve global symbol");
      } else {
        llvm::report_fatal_error("could not retrieve global symbol");
      }
    }

    addDeclareAttr(builder, globalOp.getOperation(), clause);
    auto crtPos = builder.saveInsertionPoint();
    modBuilder.setInsertionPointAfter(globalOp);
    if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(globalOp.getType()))) {
      createDeclareGlobalOp<mlir::acc::GlobalConstructorOp, mlir::acc::CopyinOp,
                            mlir::acc::DeclareEnterOp, ExitOp>(
          modBuilder, builder, operandLocation, globalOp, clause,
          declareGlobalCtorName.str(), /*implicit=*/true, asFortran);
      createDeclareAllocFunc<EntryOp>(modBuilder, builder, operandLocation,
                                      globalOp, clause);
      if constexpr (!std::is_same_v<EntryOp, ExitOp>)
        createDeclareDeallocFunc<ExitOp>(modBuilder, builder, operandLocation,
                                         globalOp, clause);
    } else {
      createDeclareGlobalOp<mlir::acc::GlobalConstructorOp, EntryOp,
                            mlir::acc::DeclareEnterOp, ExitOp>(
          modBuilder, builder, operandLocation, globalOp, clause,
          declareGlobalCtorName.str(), /*implicit=*/false, asFortran);
    }
    if constexpr (!std::is_same_v<EntryOp, ExitOp>) {
      createDeclareGlobalOp<mlir::acc::GlobalDestructorOp,
                            mlir::acc::GetDevicePtrOp, mlir::acc::DeclareExitOp,
                            ExitOp>(
          modBuilder, builder, operandLocation, globalOp, clause,
          declareGlobalDtorName.str(), /*implicit=*/false, asFortran);
    }
    builder.restoreInsertionPoint(crtPos);
  };
  for (const auto &accObject : accObjectList.v) {
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name =
                      Fortran::semantics::getDesignatorNameIfDataRef(
                          designator)) {
                genCtors(operandLocation, *name->symbol);
              }
            },
            [&](const Fortran::parser::Name &name) {
              if (const auto *symbol = name.symbol) {
                if (symbol
                        ->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
                  genCtors(operandLocation, *symbol);
                } else {
                  TODO(operandLocation,
                       "OpenACC Global Ctor from parser::Name");
                }
              }
            }},
        accObject.u);
  }
}

template <typename Clause, typename EntryOp, typename ExitOp>
static void
genGlobalCtorsWithModifier(Fortran::lower::AbstractConverter &converter,
                           mlir::OpBuilder &modBuilder, const Clause *x,
                           Fortran::parser::AccDataModifier::Modifier mod,
                           const mlir::acc::DataClause clause,
                           const mlir::acc::DataClause clauseWithModifier) {
  const Fortran::parser::AccObjectListWithModifier &listWithModifier = x->v;
  const auto &accObjectList =
      std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
  const auto &modifier =
      std::get<std::optional<Fortran::parser::AccDataModifier>>(
          listWithModifier.t);
  mlir::acc::DataClause dataClause =
      (modifier && (*modifier).v == mod) ? clauseWithModifier : clause;
  genGlobalCtors<EntryOp, ExitOp>(converter, modBuilder, accObjectList,
                                  dataClause);
}

static void
genDeclareInFunction(Fortran::lower::AbstractConverter &converter,
                     Fortran::semantics::SemanticsContext &semanticsContext,
                     Fortran::lower::StatementContext &openAccCtx,
                     mlir::Location loc,
                     const Fortran::parser::AccClauseList &accClauseList) {
  llvm::SmallVector<mlir::Value> dataClauseOperands, copyEntryOperands,
      copyinEntryOperands, createEntryOperands, copyoutEntryOperands,
      presentEntryOperands, deviceResidentEntryOperands;
  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *copyClause =
            std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::CopyinOp,
                                      mlir::acc::CopyoutOp>(
          copyClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copy,
          /*structured=*/true, /*implicit=*/false);
      copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                               dataClauseOperands.end());
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          createClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_create,
          /*structured=*/true, /*implicit=*/false);
      createEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::PresentOp,
                                      mlir::acc::DeleteOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true, /*implicit=*/false);
      presentEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperationsWithModifier<mlir::acc::CopyinOp,
                                                  mlir::acc::DeleteOp>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly);
      copyinEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyoutClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::CreateOp,
                                      mlir::acc::CopyoutOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copyout,
          /*structured=*/true, /*implicit=*/false);
      copyoutEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *devicePtrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDeclareDataOperandOperations<mlir::acc::DevicePtrOp,
                                      mlir::acc::DevicePtrOp>(
          devicePtrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *linkClause =
                   std::get_if<Fortran::parser::AccClause::Link>(&clause.u)) {
      genDeclareDataOperandOperations<mlir::acc::DeclareLinkOp,
                                      mlir::acc::DeclareLinkOp>(
          linkClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_declare_link,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *deviceResidentClause =
                   std::get_if<Fortran::parser::AccClause::DeviceResident>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::DeclareDeviceResidentOp,
                                      mlir::acc::DeleteOp>(
          deviceResidentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands,
          mlir::acc::DataClause::acc_declare_device_resident,
          /*structured=*/true, /*implicit=*/false);
      deviceResidentEntryOperands.append(
          dataClauseOperands.begin() + crtDataStart, dataClauseOperands.end());
    } else {
      mlir::Location clauseLocation = converter.genLocation(clause.source);
      TODO(clauseLocation, "clause on declare directive");
    }
  }

  mlir::func::FuncOp funcOp = builder.getFunction();
  auto ops = funcOp.getOps<mlir::acc::DeclareEnterOp>();
  mlir::Value declareToken;
  if (ops.empty()) {
    declareToken = builder.create<mlir::acc::DeclareEnterOp>(
        loc, mlir::acc::DeclareTokenType::get(builder.getContext()),
        dataClauseOperands);
  } else {
    auto declareOp = *ops.begin();
    auto newDeclareOp = builder.create<mlir::acc::DeclareEnterOp>(
        loc, mlir::acc::DeclareTokenType::get(builder.getContext()),
        declareOp.getDataClauseOperands());
    newDeclareOp.getDataClauseOperandsMutable().append(dataClauseOperands);
    declareToken = newDeclareOp.getToken();
    declareOp.erase();
  }

  openAccCtx.attachCleanup([&builder, loc, createEntryOperands,
                            copyEntryOperands, copyinEntryOperands,
                            copyoutEntryOperands, presentEntryOperands,
                            deviceResidentEntryOperands, declareToken]() {
    llvm::SmallVector<mlir::Value> operands;
    operands.append(createEntryOperands);
    operands.append(deviceResidentEntryOperands);
    operands.append(copyEntryOperands);
    operands.append(copyinEntryOperands);
    operands.append(copyoutEntryOperands);
    operands.append(presentEntryOperands);

    mlir::func::FuncOp funcOp = builder.getFunction();
    auto ops = funcOp.getOps<mlir::acc::DeclareExitOp>();
    if (ops.empty()) {
      builder.create<mlir::acc::DeclareExitOp>(loc, declareToken, operands);
    } else {
      auto declareOp = *ops.begin();
      declareOp.getDataClauseOperandsMutable().append(operands);
    }

    genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
        builder, createEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::DeclareDeviceResidentOp,
                          mlir::acc::DeleteOp>(
        builder, deviceResidentEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
        builder, copyEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::DeleteOp>(
        builder, copyinEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
        builder, copyoutEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::PresentOp, mlir::acc::DeleteOp>(
        builder, presentEntryOperands, /*structured=*/true);
  });
}

static void
genDeclareInModule(Fortran::lower::AbstractConverter &converter,
                   mlir::ModuleOp moduleOp,
                   const Fortran::parser::AccClauseList &accClauseList) {
  mlir::OpBuilder modBuilder(moduleOp.getBodyRegion());
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *createClause =
            std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          createClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genGlobalCtors<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
          converter, modBuilder, accObjectList,
          mlir::acc::DataClause::acc_create);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genGlobalCtorsWithModifier<Fortran::parser::AccClause::Copyin,
                                 mlir::acc::CopyinOp, mlir::acc::DeleteOp>(
          converter, modBuilder, copyinClause,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly);
    } else if (const auto *deviceResidentClause =
                   std::get_if<Fortran::parser::AccClause::DeviceResident>(
                       &clause.u)) {
      genGlobalCtors<mlir::acc::DeclareDeviceResidentOp, mlir::acc::DeleteOp>(
          converter, modBuilder, deviceResidentClause->v,
          mlir::acc::DataClause::acc_declare_device_resident);
    } else if (const auto *linkClause =
                   std::get_if<Fortran::parser::AccClause::Link>(&clause.u)) {
      genGlobalCtors<mlir::acc::DeclareLinkOp, mlir::acc::DeclareLinkOp>(
          converter, modBuilder, linkClause->v,
          mlir::acc::DataClause::acc_declare_link);
    } else {
      llvm::report_fatal_error("unsupported clause on DECLARE directive");
    }
  }
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   Fortran::semantics::SemanticsContext &semanticsContext,
                   Fortran::lower::StatementContext &openAccCtx,
                   const Fortran::parser::OpenACCStandaloneDeclarativeConstruct
                       &declareConstruct) {

  const auto &declarativeDir =
      std::get<Fortran::parser::AccDeclarativeDirective>(declareConstruct.t);
  mlir::Location directiveLocation =
      converter.genLocation(declarativeDir.source);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(declareConstruct.t);

  if (declarativeDir.v == llvm::acc::Directive::ACCD_declare) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    auto moduleOp =
        builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
    auto funcOp =
        builder.getBlock()->getParent()->getParentOfType<mlir::func::FuncOp>();
    if (funcOp)
      genDeclareInFunction(converter, semanticsContext, openAccCtx,
                           directiveLocation, accClauseList);
    else if (moduleOp)
      genDeclareInModule(converter, moduleOp, accClauseList);
    return;
  }
  llvm_unreachable("unsupported declarative directive");
}

static bool hasDeviceType(llvm::SmallVector<mlir::Attribute> &arrayAttr,
                          mlir::acc::DeviceType deviceType) {
  for (auto attr : arrayAttr) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
    if (deviceTypeAttr.getValue() == deviceType)
      return true;
  }
  return false;
}

template <typename RetTy, typename AttrTy>
static std::optional<RetTy>
getAttributeValueByDeviceType(llvm::SmallVector<mlir::Attribute> &attributes,
                              llvm::SmallVector<mlir::Attribute> &deviceTypes,
                              mlir::acc::DeviceType deviceType) {
  assert(attributes.size() == deviceTypes.size() &&
         "expect same number of attributes");
  for (auto it : llvm::enumerate(deviceTypes)) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(it.value());
    if (deviceTypeAttr.getValue() == deviceType) {
      if constexpr (std::is_same_v<mlir::StringAttr, AttrTy>) {
        auto strAttr = mlir::dyn_cast<AttrTy>(attributes[it.index()]);
        return strAttr.getValue();
      } else if constexpr (std::is_same_v<mlir::IntegerAttr, AttrTy>) {
        auto intAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(attributes[it.index()]);
        return intAttr.getInt();
      }
    }
  }
  return std::nullopt;
}

static bool compareDeviceTypeInfo(
    mlir::acc::RoutineOp op,
    llvm::SmallVector<mlir::Attribute> &bindNameArrayAttr,
    llvm::SmallVector<mlir::Attribute> &bindNameDeviceTypeArrayAttr,
    llvm::SmallVector<mlir::Attribute> &gangArrayAttr,
    llvm::SmallVector<mlir::Attribute> &gangDimArrayAttr,
    llvm::SmallVector<mlir::Attribute> &gangDimDeviceTypeArrayAttr,
    llvm::SmallVector<mlir::Attribute> &seqArrayAttr,
    llvm::SmallVector<mlir::Attribute> &workerArrayAttr,
    llvm::SmallVector<mlir::Attribute> &vectorArrayAttr) {
  for (uint32_t dtypeInt = 0;
       dtypeInt != mlir::acc::getMaxEnumValForDeviceType(); ++dtypeInt) {
    auto dtype = static_cast<mlir::acc::DeviceType>(dtypeInt);
    if (op.getBindNameValue(dtype) !=
        getAttributeValueByDeviceType<llvm::StringRef, mlir::StringAttr>(
            bindNameArrayAttr, bindNameDeviceTypeArrayAttr, dtype))
      return false;
    if (op.hasGang(dtype) != hasDeviceType(gangArrayAttr, dtype))
      return false;
    if (op.getGangDimValue(dtype) !=
        getAttributeValueByDeviceType<int64_t, mlir::IntegerAttr>(
            gangDimArrayAttr, gangDimDeviceTypeArrayAttr, dtype))
      return false;
    if (op.hasSeq(dtype) != hasDeviceType(seqArrayAttr, dtype))
      return false;
    if (op.hasWorker(dtype) != hasDeviceType(workerArrayAttr, dtype))
      return false;
    if (op.hasVector(dtype) != hasDeviceType(vectorArrayAttr, dtype))
      return false;
  }
  return true;
}

static void attachRoutineInfo(mlir::func::FuncOp func,
                              mlir::SymbolRefAttr routineAttr) {
  llvm::SmallVector<mlir::SymbolRefAttr> routines;
  if (func.getOperation()->hasAttr(mlir::acc::getRoutineInfoAttrName())) {
    auto routineInfo =
        func.getOperation()->getAttrOfType<mlir::acc::RoutineInfoAttr>(
            mlir::acc::getRoutineInfoAttrName());
    routines.append(routineInfo.getAccRoutines().begin(),
                    routineInfo.getAccRoutines().end());
  }
  routines.push_back(routineAttr);
  func.getOperation()->setAttr(
      mlir::acc::getRoutineInfoAttrName(),
      mlir::acc::RoutineInfoAttr::get(func.getContext(), routines));
}

static mlir::ArrayAttr
getArrayAttrOrNull(fir::FirOpBuilder &builder,
                   llvm::SmallVector<mlir::Attribute> &attributes) {
  if (attributes.empty()) {
    return nullptr;
  } else {
    return builder.getArrayAttr(attributes);
  }
}

void createOpenACCRoutineConstruct(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::ModuleOp mod, mlir::func::FuncOp funcOp, std::string funcName,
    bool hasNohost, llvm::SmallVector<mlir::Attribute> &bindNames,
    llvm::SmallVector<mlir::Attribute> &bindNameDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &gangDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &gangDimValues,
    llvm::SmallVector<mlir::Attribute> &gangDimDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &seqDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &workerDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &vectorDeviceTypes) {

  for (auto routineOp : mod.getOps<mlir::acc::RoutineOp>()) {
    if (routineOp.getFuncName().getLeafReference().str().compare(funcName) ==
        0) {
      // If the routine is already specified with the same clauses, just skip
      // the operation creation.
      if (compareDeviceTypeInfo(routineOp, bindNames, bindNameDeviceTypes,
                                gangDeviceTypes, gangDimValues,
                                gangDimDeviceTypes, seqDeviceTypes,
                                workerDeviceTypes, vectorDeviceTypes) &&
          routineOp.getNohost() == hasNohost)
        return;
      mlir::emitError(loc, "Routine already specified with different clauses");
    }
  }
  std::stringstream routineOpName;
  routineOpName << accRoutinePrefix.str() << routineCounter++;
  std::string routineOpStr = routineOpName.str();
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  modBuilder.create<mlir::acc::RoutineOp>(
      loc, routineOpStr,
      mlir::SymbolRefAttr::get(builder.getContext(), funcName),
      getArrayAttrOrNull(builder, bindNames),
      getArrayAttrOrNull(builder, bindNameDeviceTypes),
      getArrayAttrOrNull(builder, workerDeviceTypes),
      getArrayAttrOrNull(builder, vectorDeviceTypes),
      getArrayAttrOrNull(builder, seqDeviceTypes), hasNohost,
      /*implicit=*/false, getArrayAttrOrNull(builder, gangDeviceTypes),
      getArrayAttrOrNull(builder, gangDimValues),
      getArrayAttrOrNull(builder, gangDimDeviceTypes));

  attachRoutineInfo(funcOp, builder.getSymbolRefAttr(routineOpStr));
}

static void interpretRoutineDeviceInfo(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::OpenACCRoutineDeviceTypeInfo &dinfo,
    llvm::SmallVector<mlir::Attribute> &seqDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &vectorDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &workerDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &bindNameDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &bindNames,
    llvm::SmallVector<mlir::Attribute> &gangDeviceTypes,
    llvm::SmallVector<mlir::Attribute> &gangDimValues,
    llvm::SmallVector<mlir::Attribute> &gangDimDeviceTypes) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto getDeviceTypeAttr = [&]() -> mlir::Attribute {
    auto context = builder.getContext();
    auto value = getDeviceType(dinfo.dType());
    return mlir::acc::DeviceTypeAttr::get(context, value);
  };
  if (dinfo.isSeq()) {
    seqDeviceTypes.push_back(getDeviceTypeAttr());
  }
  if (dinfo.isVector()) {
    vectorDeviceTypes.push_back(getDeviceTypeAttr());
  }
  if (dinfo.isWorker()) {
    workerDeviceTypes.push_back(getDeviceTypeAttr());
  }
  if (dinfo.isGang()) {
    unsigned gangDim = dinfo.gangDim();
    auto deviceType = getDeviceTypeAttr();
    if (!gangDim) {
      gangDeviceTypes.push_back(deviceType);
    } else {
      gangDimValues.push_back(
          builder.getIntegerAttr(builder.getI64Type(), gangDim));
      gangDimDeviceTypes.push_back(deviceType);
    }
  }
  if (dinfo.bindNameOpt().has_value()) {
    const auto &bindName = dinfo.bindNameOpt().value();
    mlir::Attribute bindNameAttr;
    if (const auto &bindStr{std::get_if<std::string>(&bindName)}) {
      bindNameAttr = builder.getStringAttr(*bindStr);
    } else if (const auto &bindSym{
                   std::get_if<Fortran::semantics::SymbolRef>(&bindName)}) {
      bindNameAttr = builder.getStringAttr(converter.mangleName(*bindSym));
    } else {
      llvm_unreachable("Unsupported bind name type");
    }
    bindNames.push_back(bindNameAttr);
    bindNameDeviceTypes.push_back(getDeviceTypeAttr());
  }
}

void Fortran::lower::genOpenACCRoutineConstruct(
    Fortran::lower::AbstractConverter &converter, mlir::ModuleOp mod,
    mlir::func::FuncOp funcOp,
    const std::vector<Fortran::semantics::OpenACCRoutineInfo> &routineInfos) {
  CHECK(funcOp && "Expected a valid function operation");
  mlir::Location loc{funcOp.getLoc()};
  std::string funcName{funcOp.getName()};

  // Collect the routine clauses
  bool hasNohost{false};

  llvm::SmallVector<mlir::Attribute> seqDeviceTypes, vectorDeviceTypes,
      workerDeviceTypes, bindNameDeviceTypes, bindNames, gangDeviceTypes,
      gangDimDeviceTypes, gangDimValues;

  for (const Fortran::semantics::OpenACCRoutineInfo &info : routineInfos) {
    // Device Independent Attributes
    if (info.isNohost()) {
      hasNohost = true;
    }
    // Note: Device Independent Attributes are set to the
    // none device type in `info`.
    interpretRoutineDeviceInfo(converter, info, seqDeviceTypes,
                               vectorDeviceTypes, workerDeviceTypes,
                               bindNameDeviceTypes, bindNames, gangDeviceTypes,
                               gangDimValues, gangDimDeviceTypes);

    // Device Dependent Attributes
    for (const Fortran::semantics::OpenACCRoutineDeviceTypeInfo &dinfo :
         info.deviceTypeInfos()) {
      interpretRoutineDeviceInfo(
          converter, dinfo, seqDeviceTypes, vectorDeviceTypes,
          workerDeviceTypes, bindNameDeviceTypes, bindNames, gangDeviceTypes,
          gangDimValues, gangDimDeviceTypes);
    }
  }
  createOpenACCRoutineConstruct(
      converter, loc, mod, funcOp, funcName, hasNohost, bindNames,
      bindNameDeviceTypes, gangDeviceTypes, gangDimValues, gangDimDeviceTypes,
      seqDeviceTypes, workerDeviceTypes, vectorDeviceTypes);
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {

  mlir::Location loc = converter.genLocation(atomicConstruct.source);
  Fortran::common::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::AccAtomicRead &atomicRead) {
            genAtomicRead(converter, atomicRead, loc);
          },
          [&](const Fortran::parser::AccAtomicWrite &atomicWrite) {
            genAtomicWrite(converter, atomicWrite, loc);
          },
          [&](const Fortran::parser::AccAtomicUpdate &atomicUpdate) {
            genAtomicUpdate(converter, atomicUpdate, loc);
          },
          [&](const Fortran::parser::AccAtomicCapture &atomicCapture) {
            genAtomicCapture(converter, atomicCapture, loc);
          },
      },
      atomicConstruct.u);
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto loopOp = builder.getRegion().getParentOfType<mlir::acc::LoopOp>();
  auto crtPos = builder.saveInsertionPoint();
  if (loopOp) {
    builder.setInsertionPoint(loopOp);
    Fortran::lower::StatementContext stmtCtx;
    llvm::SmallVector<mlir::Value> cacheOperands;
    const Fortran::parser::AccObjectListWithModifier &listWithModifier =
        std::get<Fortran::parser::AccObjectListWithModifier>(cacheConstruct.t);
    const auto &accObjectList =
        std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
    const auto &modifier =
        std::get<std::optional<Fortran::parser::AccDataModifier>>(
            listWithModifier.t);

    mlir::acc::DataClause dataClause = mlir::acc::DataClause::acc_cache;
    if (modifier &&
        (*modifier).v == Fortran::parser::AccDataModifier::Modifier::ReadOnly)
      dataClause = mlir::acc::DataClause::acc_cache_readonly;
    genDataOperandOperations<mlir::acc::CacheOp>(
        accObjectList, converter, semanticsContext, stmtCtx, cacheOperands,
        dataClause,
        /*structured=*/true, /*implicit=*/false,
        /*async=*/{}, /*asyncDeviceTypes=*/{}, /*asyncOnlyDeviceTypes=*/{},
        /*setDeclareAttr*/ false);
    loopOp.getCacheOperandsMutable().append(cacheOperands);
  } else {
    llvm::report_fatal_error(
        "could not find loop to attach OpenACC cache information.");
  }
  builder.restoreInsertionPoint(crtPos);
}

mlir::Value Fortran::lower::genOpenACCConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCConstruct &accConstruct) {

  mlir::Value exitCond;
  Fortran::common::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
            genACC(converter, semanticsContext, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenACCCombinedConstruct
                  &combinedConstruct) {
            genACC(converter, semanticsContext, eval, combinedConstruct);
          },
          [&](const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {
            exitCond = genACC(converter, semanticsContext, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenACCStandaloneConstruct
                  &standaloneConstruct) {
            genACC(converter, semanticsContext, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
            genACC(converter, semanticsContext, cacheConstruct);
          },
          [&](const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {
            genACC(converter, waitConstruct);
          },
          [&](const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {
            genACC(converter, eval, atomicConstruct);
          },
          [&](const Fortran::parser::OpenACCEndConstruct &) {
            // No op
          },
      },
      accConstruct.u);
  return exitCond;
}

void Fortran::lower::genOpenACCDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &openAccCtx,
    const Fortran::parser::OpenACCDeclarativeConstruct &accDeclConstruct) {

  Fortran::common::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCStandaloneDeclarativeConstruct
                  &standaloneDeclarativeConstruct) {
            genACC(converter, semanticsContext, openAccCtx,
                   standaloneDeclarativeConstruct);
          },
          [&](const Fortran::parser::OpenACCRoutineConstruct &x) {},
      },
      accDeclConstruct.u);
}

void Fortran::lower::attachDeclarePostAllocAction(
    AbstractConverter &converter, fir::FirOpBuilder &builder,
    const Fortran::semantics::Symbol &sym) {
  std::stringstream fctName;
  fctName << converter.mangleName(sym) << declarePostAllocSuffix.str();
  mlir::Operation *op = &builder.getInsertionBlock()->back();

  if (auto resOp = mlir::dyn_cast<fir::ResultOp>(*op)) {
    assert(resOp.getOperands().size() == 0 &&
           "expect only fir.result op with no operand");
    op = op->getPrevNode();
  }
  assert(op && "expect operation to attach the post allocation action");

  if (op->hasAttr(mlir::acc::getDeclareActionAttrName())) {
    auto attr = op->getAttrOfType<mlir::acc::DeclareActionAttr>(
        mlir::acc::getDeclareActionAttrName());
    op->setAttr(mlir::acc::getDeclareActionAttrName(),
                mlir::acc::DeclareActionAttr::get(
                    builder.getContext(), attr.getPreAlloc(),
                    /*postAlloc=*/builder.getSymbolRefAttr(fctName.str()),
                    attr.getPreDealloc(), attr.getPostDealloc()));
  } else {
    op->setAttr(mlir::acc::getDeclareActionAttrName(),
                mlir::acc::DeclareActionAttr::get(
                    builder.getContext(),
                    /*preAlloc=*/{},
                    /*postAlloc=*/builder.getSymbolRefAttr(fctName.str()),
                    /*preDealloc=*/{}, /*postDealloc=*/{}));
  }
}

void Fortran::lower::attachDeclarePreDeallocAction(
    AbstractConverter &converter, fir::FirOpBuilder &builder,
    mlir::Value beginOpValue, const Fortran::semantics::Symbol &sym) {
  if (!sym.test(Fortran::semantics::Symbol::Flag::AccCreate) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyIn) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyInReadOnly) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopy) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyOut) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccDeviceResident))
    return;

  std::stringstream fctName;
  fctName << converter.mangleName(sym) << declarePreDeallocSuffix.str();

  auto *op = beginOpValue.getDefiningOp();
  if (op->hasAttr(mlir::acc::getDeclareActionAttrName())) {
    auto attr = op->getAttrOfType<mlir::acc::DeclareActionAttr>(
        mlir::acc::getDeclareActionAttrName());
    op->setAttr(mlir::acc::getDeclareActionAttrName(),
                mlir::acc::DeclareActionAttr::get(
                    builder.getContext(), attr.getPreAlloc(),
                    attr.getPostAlloc(),
                    /*preDealloc=*/builder.getSymbolRefAttr(fctName.str()),
                    attr.getPostDealloc()));
  } else {
    op->setAttr(mlir::acc::getDeclareActionAttrName(),
                mlir::acc::DeclareActionAttr::get(
                    builder.getContext(),
                    /*preAlloc=*/{}, /*postAlloc=*/{},
                    /*preDealloc=*/builder.getSymbolRefAttr(fctName.str()),
                    /*postDealloc=*/{}));
  }
}

void Fortran::lower::attachDeclarePostDeallocAction(
    AbstractConverter &converter, fir::FirOpBuilder &builder,
    const Fortran::semantics::Symbol &sym) {
  if (!sym.test(Fortran::semantics::Symbol::Flag::AccCreate) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyIn) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyInReadOnly) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopy) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyOut) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccDeviceResident))
    return;

  std::stringstream fctName;
  fctName << converter.mangleName(sym) << declarePostDeallocSuffix.str();
  mlir::Operation *op = &builder.getInsertionBlock()->back();
  if (auto resOp = mlir::dyn_cast<fir::ResultOp>(*op)) {
    assert(resOp.getOperands().size() == 0 &&
           "expect only fir.result op with no operand");
    op = op->getPrevNode();
  }
  assert(op && "expect operation to attach the post deallocation action");
  if (op->hasAttr(mlir::acc::getDeclareActionAttrName())) {
    auto attr = op->getAttrOfType<mlir::acc::DeclareActionAttr>(
        mlir::acc::getDeclareActionAttrName());
    op->setAttr(mlir::acc::getDeclareActionAttrName(),
                mlir::acc::DeclareActionAttr::get(
                    builder.getContext(), attr.getPreAlloc(),
                    attr.getPostAlloc(), attr.getPreDealloc(),
                    /*postDealloc=*/builder.getSymbolRefAttr(fctName.str())));
  } else {
    op->setAttr(mlir::acc::getDeclareActionAttrName(),
                mlir::acc::DeclareActionAttr::get(
                    builder.getContext(),
                    /*preAlloc=*/{}, /*postAlloc=*/{}, /*preDealloc=*/{},
                    /*postDealloc=*/builder.getSymbolRefAttr(fctName.str())));
  }
}

void Fortran::lower::genOpenACCTerminator(fir::FirOpBuilder &builder,
                                          mlir::Operation *op,
                                          mlir::Location loc) {
  if (mlir::isa<mlir::acc::ParallelOp, mlir::acc::LoopOp>(op))
    builder.create<mlir::acc::YieldOp>(loc);
  else
    builder.create<mlir::acc::TerminatorOp>(loc);
}

bool Fortran::lower::isInOpenACCLoop(fir::FirOpBuilder &builder) {
  if (builder.getBlock()->getParent()->getParentOfType<mlir::acc::LoopOp>())
    return true;
  return false;
}

void Fortran::lower::setInsertionPointAfterOpenACCLoopIfInside(
    fir::FirOpBuilder &builder) {
  if (auto loopOp =
          builder.getBlock()->getParent()->getParentOfType<mlir::acc::LoopOp>())
    builder.setInsertionPointAfter(loopOp);
}

void Fortran::lower::genEarlyReturnInOpenACCLoop(fir::FirOpBuilder &builder,
                                                 mlir::Location loc) {
  mlir::Value yieldValue =
      builder.createIntegerConstant(loc, builder.getI1Type(), 1);
  builder.create<mlir::acc::YieldOp>(loc, yieldValue);
}

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::AccClauseList &clauseList) {
  for (const Fortran::parser::AccClause &clause : clauseList.v) {
    if (const auto *collapseClause =
            std::get_if<Fortran::parser::AccClause::Collapse>(&clause.u)) {
      const parser::AccCollapseArg &arg = collapseClause->v;
      const auto &collapseValue{std::get<parser::ScalarIntConstantExpr>(arg.t)};
      return *Fortran::semantics::GetIntValue(collapseValue);
    }
  }
  return 1;
}
