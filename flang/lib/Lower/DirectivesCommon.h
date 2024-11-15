//===-- Lower/DirectivesCommon.h --------------------------------*- C++ -*-===//
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
///
/// A location to place directive utilities shared across multiple lowering
/// files, e.g. utilities shared in OpenMP and OpenACC. The header file can
/// be used for both declarations and templated/inline implementations
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_DIRECTIVES_COMMON_H
#define FORTRAN_LOWER_DIRECTIVES_COMMON_H

#include "flang/Common/idioms.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <list>
#include <type_traits>

namespace Fortran {
namespace lower {

/// Information gathered to generate bounds operation and data entry/exit
/// operations.
struct AddrAndBoundsInfo {
  explicit AddrAndBoundsInfo() {}
  explicit AddrAndBoundsInfo(mlir::Value addr, mlir::Value rawInput)
      : addr(addr), rawInput(rawInput) {}
  explicit AddrAndBoundsInfo(mlir::Value addr, mlir::Value rawInput,
                             mlir::Value isPresent)
      : addr(addr), rawInput(rawInput), isPresent(isPresent) {}
  explicit AddrAndBoundsInfo(mlir::Value addr, mlir::Value rawInput,
                             mlir::Value isPresent, mlir::Type boxType)
      : addr(addr), rawInput(rawInput), isPresent(isPresent), boxType(boxType) {
  }
  mlir::Value addr = nullptr;
  mlir::Value rawInput = nullptr;
  mlir::Value isPresent = nullptr;
  mlir::Type boxType = nullptr;
  void dump(llvm::raw_ostream &os) {
    os << "AddrAndBoundsInfo addr: " << addr << "\n";
    os << "AddrAndBoundsInfo rawInput: " << rawInput << "\n";
    os << "AddrAndBoundsInfo isPresent: " << isPresent << "\n";
    os << "AddrAndBoundsInfo boxType: " << boxType << "\n";
  }
};

/// Populates \p hint and \p memoryOrder with appropriate clause information
/// if present on atomic construct.
static inline void genOmpAtomicHintAndMemoryOrderClauses(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::OmpAtomicClauseList &clauseList,
    mlir::IntegerAttr &hint,
    mlir::omp::ClauseMemoryOrderKindAttr &memoryOrder) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  for (const Fortran::parser::OmpAtomicClause &clause : clauseList.v) {
    if (const auto *ompClause =
            std::get_if<Fortran::parser::OmpClause>(&clause.u)) {
      if (const auto *hintClause =
              std::get_if<Fortran::parser::OmpClause::Hint>(&ompClause->u)) {
        const auto *expr = Fortran::semantics::GetExpr(hintClause->v);
        uint64_t hintExprValue = *Fortran::evaluate::ToInt64(*expr);
        hint = firOpBuilder.getI64IntegerAttr(hintExprValue);
      }
    } else if (const auto *ompMemoryOrderClause =
                   std::get_if<Fortran::parser::OmpMemoryOrderClause>(
                       &clause.u)) {
      if (std::get_if<Fortran::parser::OmpClause::Acquire>(
              &ompMemoryOrderClause->v.u)) {
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(),
            mlir::omp::ClauseMemoryOrderKind::Acquire);
      } else if (std::get_if<Fortran::parser::OmpClause::Relaxed>(
                     &ompMemoryOrderClause->v.u)) {
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(),
            mlir::omp::ClauseMemoryOrderKind::Relaxed);
      } else if (std::get_if<Fortran::parser::OmpClause::SeqCst>(
                     &ompMemoryOrderClause->v.u)) {
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(),
            mlir::omp::ClauseMemoryOrderKind::Seq_cst);
      } else if (std::get_if<Fortran::parser::OmpClause::Release>(
                     &ompMemoryOrderClause->v.u)) {
        memoryOrder = mlir::omp::ClauseMemoryOrderKindAttr::get(
            firOpBuilder.getContext(),
            mlir::omp::ClauseMemoryOrderKind::Release);
      }
    }
  }
}

template <typename AtomicListT>
static void processOmpAtomicTODO(mlir::Type elementType,
                                 [[maybe_unused]] mlir::Location loc) {
  if (!elementType)
    return;
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    // Based on assertion for supported element types in OMPIRBuilder.cpp
    // createAtomicRead
    mlir::Type unwrappedEleTy = fir::unwrapRefType(elementType);
    bool supportedAtomicType = fir::isa_trivial(unwrappedEleTy);
    if (!supportedAtomicType)
      TODO(loc, "Unsupported atomic type");
  }
}

/// Used to generate atomic.read operation which is created in existing
/// location set by builder.
template <typename AtomicListT>
static inline void genOmpAccAtomicCaptureStatement(
    Fortran::lower::AbstractConverter &converter, mlir::Value fromAddress,
    mlir::Value toAddress,
    [[maybe_unused]] const AtomicListT *leftHandClauseList,
    [[maybe_unused]] const AtomicListT *rightHandClauseList,
    mlir::Type elementType, mlir::Location loc) {
  // Generate `atomic.read` operation for atomic assigment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  processOmpAtomicTODO<AtomicListT>(elementType, loc);

  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    // If no hint clause is specified, the effect is as if
    // hint(omp_sync_hint_none) had been specified.
    mlir::IntegerAttr hint = nullptr;

    mlir::omp::ClauseMemoryOrderKindAttr memoryOrder = nullptr;
    if (leftHandClauseList)
      genOmpAtomicHintAndMemoryOrderClauses(converter, *leftHandClauseList,
                                            hint, memoryOrder);
    if (rightHandClauseList)
      genOmpAtomicHintAndMemoryOrderClauses(converter, *rightHandClauseList,
                                            hint, memoryOrder);
    firOpBuilder.create<mlir::omp::AtomicReadOp>(
        loc, fromAddress, toAddress, mlir::TypeAttr::get(elementType), hint,
        memoryOrder);
  } else {
    firOpBuilder.create<mlir::acc::AtomicReadOp>(
        loc, fromAddress, toAddress, mlir::TypeAttr::get(elementType));
  }
}

/// Used to generate atomic.write operation which is created in existing
/// location set by builder.
template <typename AtomicListT>
static inline void genOmpAccAtomicWriteStatement(
    Fortran::lower::AbstractConverter &converter, mlir::Value lhsAddr,
    mlir::Value rhsExpr, [[maybe_unused]] const AtomicListT *leftHandClauseList,
    [[maybe_unused]] const AtomicListT *rightHandClauseList, mlir::Location loc,
    mlir::Value *evaluatedExprValue = nullptr) {
  // Generate `atomic.write` operation for atomic assignment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::Type varType = fir::unwrapRefType(lhsAddr.getType());
  rhsExpr = firOpBuilder.createConvert(loc, varType, rhsExpr);

  processOmpAtomicTODO<AtomicListT>(varType, loc);

  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    // If no hint clause is specified, the effect is as if
    // hint(omp_sync_hint_none) had been specified.
    mlir::IntegerAttr hint = nullptr;
    mlir::omp::ClauseMemoryOrderKindAttr memoryOrder = nullptr;
    if (leftHandClauseList)
      genOmpAtomicHintAndMemoryOrderClauses(converter, *leftHandClauseList,
                                            hint, memoryOrder);
    if (rightHandClauseList)
      genOmpAtomicHintAndMemoryOrderClauses(converter, *rightHandClauseList,
                                            hint, memoryOrder);
    firOpBuilder.create<mlir::omp::AtomicWriteOp>(loc, lhsAddr, rhsExpr, hint,
                                                  memoryOrder);
  } else {
    firOpBuilder.create<mlir::acc::AtomicWriteOp>(loc, lhsAddr, rhsExpr);
  }
}

/// Used to generate atomic.update operation which is created in existing
/// location set by builder.
template <typename AtomicListT>
static inline void genOmpAccAtomicUpdateStatement(
    Fortran::lower::AbstractConverter &converter, mlir::Value lhsAddr,
    mlir::Type varType, const Fortran::parser::Variable &assignmentStmtVariable,
    const Fortran::parser::Expr &assignmentStmtExpr,
    [[maybe_unused]] const AtomicListT *leftHandClauseList,
    [[maybe_unused]] const AtomicListT *rightHandClauseList, mlir::Location loc,
    mlir::Operation *atomicCaptureOp = nullptr) {
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
      [](std::list<parser::ActualArgSpec>::const_iterator it) {
        const auto &arg{std::get<parser::ActualArg>((*it).t)};
        const auto *parserExpr{
            std::get_if<common::Indirection<parser::Expr>>(&arg.u)};
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
          [&](const common::Indirection<parser::FunctionReference> &funcRef)
              -> void {
            const auto &args{std::get<std::list<parser::ActualArgSpec>>(
                funcRef.value().v.t)};
            std::list<parser::ActualArgSpec>::const_iterator beginIt =
                args.begin();
            std::list<parser::ActualArgSpec>::const_iterator endIt = args.end();
            const auto *exprFirst{getArgExpression(beginIt)};
            if (exprFirst && exprFirst->value().source ==
                                 assignmentStmtVariable.GetSource()) {
              // Add everything except the first
              beginIt++;
            } else {
              // Add everything except the last
              endIt--;
            }
            std::list<parser::ActualArgSpec>::const_iterator it;
            for (it = beginIt; it != endIt; it++) {
              const common::Indirection<parser::Expr> *expr =
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
  StatementContext nonAtomicStmtCtx;
  if (!nonAtomicSubExprs.empty()) {
    // Generate non atomic part before all the atomic operations.
    auto insertionPoint = firOpBuilder.saveInsertionPoint();
    if (atomicCaptureOp)
      firOpBuilder.setInsertionPoint(atomicCaptureOp);
    mlir::Value nonAtomicVal;
    for (auto *nonAtomicSubExpr : nonAtomicSubExprs) {
      nonAtomicVal = fir::getBase(converter.genExprValue(
          currentLocation, *nonAtomicSubExpr, nonAtomicStmtCtx));
      exprValueOverrides.try_emplace(nonAtomicSubExpr, nonAtomicVal);
    }
    if (atomicCaptureOp)
      firOpBuilder.restoreInsertionPoint(insertionPoint);
  }

  mlir::Operation *atomicUpdateOp = nullptr;
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    // If no hint clause is specified, the effect is as if
    // hint(omp_sync_hint_none) had been specified.
    mlir::IntegerAttr hint = nullptr;
    mlir::omp::ClauseMemoryOrderKindAttr memoryOrder = nullptr;
    if (leftHandClauseList)
      genOmpAtomicHintAndMemoryOrderClauses(converter, *leftHandClauseList,
                                            hint, memoryOrder);
    if (rightHandClauseList)
      genOmpAtomicHintAndMemoryOrderClauses(converter, *rightHandClauseList,
                                            hint, memoryOrder);
    atomicUpdateOp = firOpBuilder.create<mlir::omp::AtomicUpdateOp>(
        currentLocation, lhsAddr, hint, memoryOrder);
  } else {
    atomicUpdateOp = firOpBuilder.create<mlir::acc::AtomicUpdateOp>(
        currentLocation, lhsAddr);
  }

  processOmpAtomicTODO<AtomicListT>(varType, loc);

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
    if constexpr (std::is_same<AtomicListT,
                               Fortran::parser::OmpAtomicClauseList>()) {
      firOpBuilder.create<mlir::omp::YieldOp>(currentLocation, convertResult);
    } else {
      firOpBuilder.create<mlir::acc::YieldOp>(currentLocation, convertResult);
    }
    converter.resetExprOverrides();
  }
  firOpBuilder.setInsertionPointAfter(atomicUpdateOp);
}

/// Processes an atomic construct with write clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicWrite(Fortran::lower::AbstractConverter &converter,
                          const AtomicT &atomicWrite, mlir::Location loc) {
  const AtomicListT *rightHandClauseList = nullptr;
  const AtomicListT *leftHandClauseList = nullptr;
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    // Get the address of atomic read operands.
    rightHandClauseList = &std::get<2>(atomicWrite.t);
    leftHandClauseList = &std::get<0>(atomicWrite.t);
  }

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
  genOmpAccAtomicWriteStatement(converter, lhsAddr, rhsExpr, leftHandClauseList,
                                rightHandClauseList, loc);
}

/// Processes an atomic construct with read clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicRead(Fortran::lower::AbstractConverter &converter,
                         const AtomicT &atomicRead, mlir::Location loc) {
  const AtomicListT *rightHandClauseList = nullptr;
  const AtomicListT *leftHandClauseList = nullptr;
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    // Get the address of atomic read operands.
    rightHandClauseList = &std::get<2>(atomicRead.t);
    leftHandClauseList = &std::get<0>(atomicRead.t);
  }

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
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (fromAddress.getType() != toAddress.getType())
    fromAddress =
        builder.create<fir::ConvertOp>(loc, toAddress.getType(), fromAddress);
  genOmpAccAtomicCaptureStatement(converter, fromAddress, toAddress,
                                  leftHandClauseList, rightHandClauseList,
                                  elementType, loc);
}

/// Processes an atomic construct with update clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicUpdate(Fortran::lower::AbstractConverter &converter,
                           const AtomicT &atomicUpdate, mlir::Location loc) {
  const AtomicListT *rightHandClauseList = nullptr;
  const AtomicListT *leftHandClauseList = nullptr;
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    // Get the address of atomic read operands.
    rightHandClauseList = &std::get<2>(atomicUpdate.t);
    leftHandClauseList = &std::get<0>(atomicUpdate.t);
  }

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
  genOmpAccAtomicUpdateStatement<AtomicListT>(
      converter, lhsAddr, varType, assignmentStmtVariable, assignmentStmtExpr,
      leftHandClauseList, rightHandClauseList, loc);
}

/// Processes an atomic construct with no clause - which implies update clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAtomic(Fortran::lower::AbstractConverter &converter,
                  const AtomicT &atomicConstruct, mlir::Location loc) {
  const AtomicListT &atomicClauseList =
      std::get<AtomicListT>(atomicConstruct.t);
  const auto &assignmentStmtExpr = std::get<Fortran::parser::Expr>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicConstruct.t)
          .statement.t);
  const auto &assignmentStmtVariable = std::get<Fortran::parser::Variable>(
      std::get<Fortran::parser::Statement<Fortran::parser::AssignmentStmt>>(
          atomicConstruct.t)
          .statement.t);
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value lhsAddr = fir::getBase(converter.genExprAddr(
      *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx));
  mlir::Type varType = fir::unwrapRefType(lhsAddr.getType());
  // If atomic-clause is not present on the construct, the behaviour is as if
  // the update clause is specified (for both OpenMP and OpenACC).
  genOmpAccAtomicUpdateStatement<AtomicListT>(
      converter, lhsAddr, varType, assignmentStmtVariable, assignmentStmtExpr,
      &atomicClauseList, nullptr, loc);
}

/// Processes an atomic construct with capture clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicCapture(Fortran::lower::AbstractConverter &converter,
                            const AtomicT &atomicCapture, mlir::Location loc) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  const Fortran::parser::AssignmentStmt &stmt1 =
      std::get<typename AtomicT::Stmt1>(atomicCapture.t).v.statement;
  const Fortran::evaluate::Assignment &assign1 = *stmt1.typedAssignment->v;
  const auto &stmt1Var{std::get<Fortran::parser::Variable>(stmt1.t)};
  const auto &stmt1Expr{std::get<Fortran::parser::Expr>(stmt1.t)};
  const Fortran::parser::AssignmentStmt &stmt2 =
      std::get<typename AtomicT::Stmt2>(atomicCapture.t).v.statement;
  const Fortran::evaluate::Assignment &assign2 = *stmt2.typedAssignment->v;
  const auto &stmt2Var{std::get<Fortran::parser::Variable>(stmt2.t)};
  const auto &stmt2Expr{std::get<Fortran::parser::Expr>(stmt2.t)};

  // Pre-evaluate expressions to be used in the various operations inside
  // `atomic.capture` since it is not desirable to have anything other than
  // a `atomic.read`, `atomic.write`, or `atomic.update` operation
  // inside `atomic.capture`
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value stmt1LHSArg, stmt1RHSArg, stmt2LHSArg, stmt2RHSArg;
  mlir::Type elementType;
  // LHS evaluations are common to all combinations of `atomic.capture`
  stmt1LHSArg = fir::getBase(converter.genExprAddr(assign1.lhs, stmtCtx));
  stmt2LHSArg = fir::getBase(converter.genExprAddr(assign2.lhs, stmtCtx));

  // Operation specific RHS evaluations
  if (Fortran::semantics::checkForSingleVariableOnRHS(stmt1)) {
    // Atomic capture construct is of the form [capture-stmt, update-stmt] or
    // of the form [capture-stmt, write-stmt]
    stmt1RHSArg = fir::getBase(converter.genExprAddr(assign1.rhs, stmtCtx));
    stmt2RHSArg = fir::getBase(converter.genExprValue(assign2.rhs, stmtCtx));
  } else {
    // Atomic capture construct is of the form [update-stmt, capture-stmt]
    stmt1RHSArg = fir::getBase(converter.genExprValue(assign1.rhs, stmtCtx));
    stmt2RHSArg = fir::getBase(converter.genExprAddr(assign2.lhs, stmtCtx));
  }
  // Type information used in generation of `atomic.update` operation
  mlir::Type stmt1VarType =
      fir::getBase(converter.genExprValue(assign1.lhs, stmtCtx)).getType();
  mlir::Type stmt2VarType =
      fir::getBase(converter.genExprValue(assign2.lhs, stmtCtx)).getType();

  mlir::Operation *atomicCaptureOp = nullptr;
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    mlir::IntegerAttr hint = nullptr;
    mlir::omp::ClauseMemoryOrderKindAttr memoryOrder = nullptr;
    const AtomicListT &rightHandClauseList = std::get<2>(atomicCapture.t);
    const AtomicListT &leftHandClauseList = std::get<0>(atomicCapture.t);
    genOmpAtomicHintAndMemoryOrderClauses(converter, leftHandClauseList, hint,
                                          memoryOrder);
    genOmpAtomicHintAndMemoryOrderClauses(converter, rightHandClauseList, hint,
                                          memoryOrder);
    atomicCaptureOp =
        firOpBuilder.create<mlir::omp::AtomicCaptureOp>(loc, hint, memoryOrder);
  } else {
    atomicCaptureOp = firOpBuilder.create<mlir::acc::AtomicCaptureOp>(loc);
  }

  firOpBuilder.createBlock(&(atomicCaptureOp->getRegion(0)));
  mlir::Block &block = atomicCaptureOp->getRegion(0).back();
  firOpBuilder.setInsertionPointToStart(&block);
  if (Fortran::semantics::checkForSingleVariableOnRHS(stmt1)) {
    if (Fortran::semantics::checkForSymbolMatch(stmt2)) {
      // Atomic capture construct is of the form [capture-stmt, update-stmt]
      const Fortran::semantics::SomeExpr &fromExpr =
          *Fortran::semantics::GetExpr(stmt1Expr);
      elementType = converter.genType(fromExpr);
      genOmpAccAtomicCaptureStatement<AtomicListT>(
          converter, stmt1RHSArg, stmt1LHSArg,
          /*leftHandClauseList=*/nullptr,
          /*rightHandClauseList=*/nullptr, elementType, loc);
      genOmpAccAtomicUpdateStatement<AtomicListT>(
          converter, stmt1RHSArg, stmt2VarType, stmt2Var, stmt2Expr,
          /*leftHandClauseList=*/nullptr,
          /*rightHandClauseList=*/nullptr, loc, atomicCaptureOp);
    } else {
      // Atomic capture construct is of the form [capture-stmt, write-stmt]
      const Fortran::semantics::SomeExpr &fromExpr =
          *Fortran::semantics::GetExpr(stmt1Expr);
      elementType = converter.genType(fromExpr);
      genOmpAccAtomicCaptureStatement<AtomicListT>(
          converter, stmt1RHSArg, stmt1LHSArg,
          /*leftHandClauseList=*/nullptr,
          /*rightHandClauseList=*/nullptr, elementType, loc);
      genOmpAccAtomicWriteStatement<AtomicListT>(
          converter, stmt1RHSArg, stmt2RHSArg,
          /*leftHandClauseList=*/nullptr,
          /*rightHandClauseList=*/nullptr, loc);
    }
  } else {
    // Atomic capture construct is of the form [update-stmt, capture-stmt]
    firOpBuilder.setInsertionPointToEnd(&block);
    const Fortran::semantics::SomeExpr &fromExpr =
        *Fortran::semantics::GetExpr(stmt2Expr);
    elementType = converter.genType(fromExpr);
    genOmpAccAtomicCaptureStatement<AtomicListT>(
        converter, stmt1LHSArg, stmt2LHSArg,
        /*leftHandClauseList=*/nullptr,
        /*rightHandClauseList=*/nullptr, elementType, loc);
    firOpBuilder.setInsertionPointToStart(&block);
    genOmpAccAtomicUpdateStatement<AtomicListT>(
        converter, stmt1LHSArg, stmt1VarType, stmt1Var, stmt1Expr,
        /*leftHandClauseList=*/nullptr,
        /*rightHandClauseList=*/nullptr, loc, atomicCaptureOp);
  }
  firOpBuilder.setInsertionPointToEnd(&block);
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  } else {
    firOpBuilder.create<mlir::acc::TerminatorOp>(loc);
  }
  firOpBuilder.setInsertionPointToStart(&block);
}

/// Create empty blocks for the current region.
/// These blocks replace blocks parented to an enclosing region.
template <typename... TerminatorOps>
void createEmptyRegionBlocks(
    fir::FirOpBuilder &builder,
    std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
  mlir::Region *region = &builder.getRegion();
  for (Fortran::lower::pft::Evaluation &eval : evaluationList) {
    if (eval.block) {
      if (eval.block->empty()) {
        eval.block->erase();
        eval.block = builder.createBlock(region);
      } else {
        [[maybe_unused]] mlir::Operation &terminatorOp = eval.block->back();
        assert(mlir::isa<TerminatorOps...>(terminatorOp) &&
               "expected terminator op");
      }
    }
    if (!eval.isDirective() && eval.hasNestedEvaluations())
      createEmptyRegionBlocks<TerminatorOps...>(builder,
                                                eval.getNestedEvaluations());
  }
}

inline AddrAndBoundsInfo
getDataOperandBaseAddr(Fortran::lower::AbstractConverter &converter,
                       fir::FirOpBuilder &builder,
                       Fortran::lower::SymbolRef sym, mlir::Location loc) {
  mlir::Value symAddr = converter.getSymbolAddress(sym);
  mlir::Value rawInput = symAddr;
  if (auto declareOp =
          mlir::dyn_cast_or_null<hlfir::DeclareOp>(symAddr.getDefiningOp())) {
    symAddr = declareOp.getResults()[0];
    rawInput = declareOp.getResults()[1];
  }

  // TODO: Might need revisiting to handle for non-shared clauses
  if (!symAddr) {
    if (const auto *details =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>()) {
      symAddr = converter.getSymbolAddress(details->symbol());
      rawInput = symAddr;
    }
  }

  if (!symAddr)
    llvm::report_fatal_error("could not retrieve symbol address");

  mlir::Value isPresent;
  if (Fortran::semantics::IsOptional(sym))
    isPresent =
        builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), rawInput);

  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(
          fir::unwrapRefType(symAddr.getType()))) {
    if (mlir::isa<fir::RecordType>(boxTy.getEleTy()))
      TODO(loc, "derived type");

    // In case of a box reference, load it here to get the box value.
    // This is preferrable because then the same box value can then be used for
    // all address/dimension retrievals. For Fortran optional though, leave
    // the load generation for later so it can be done in the appropriate
    // if branches.
    if (mlir::isa<fir::ReferenceType>(symAddr.getType()) &&
        !Fortran::semantics::IsOptional(sym)) {
      mlir::Value addr = builder.create<fir::LoadOp>(loc, symAddr);
      return AddrAndBoundsInfo(addr, rawInput, isPresent, boxTy);
    }

    return AddrAndBoundsInfo(symAddr, rawInput, isPresent, boxTy);
  }
  return AddrAndBoundsInfo(symAddr, rawInput, isPresent);
}

template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
gatherBoundsOrBoundValues(fir::FirOpBuilder &builder, mlir::Location loc,
                          fir::ExtendedValue dataExv, mlir::Value box,
                          bool collectValuesOnly = false) {
  assert(box && "box must exist");
  llvm::SmallVector<mlir::Value> values;
  mlir::Value byteStride;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (unsigned dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value d = builder.createIntegerConstant(loc, idxTy, dim);
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    auto dimInfo =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, d);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value ub =
        builder.create<mlir::arith::SubIOp>(loc, dimInfo.getExtent(), one);
    if (dim == 0) // First stride is the element size.
      byteStride = dimInfo.getByteStride();
    if (collectValuesOnly) {
      values.push_back(lb);
      values.push_back(ub);
      values.push_back(dimInfo.getExtent());
      values.push_back(byteStride);
      values.push_back(baseLb);
    } else {
      mlir::Value bound = builder.create<BoundsOp>(
          loc, boundTy, lb, ub, dimInfo.getExtent(), byteStride, true, baseLb);
      values.push_back(bound);
    }
    // Compute the stride for the next dimension.
    byteStride = builder.create<mlir::arith::MulIOp>(loc, byteStride,
                                                     dimInfo.getExtent());
  }
  return values;
}

/// Generate the bounds operation from the descriptor information.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBoundsOpsFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                    fir::ExtendedValue dataExv,
                    Fortran::lower::AddrAndBoundsInfo &info) {
  llvm::SmallVector<mlir::Value> bounds;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();

  assert(mlir::isa<fir::BaseBoxType>(info.boxType) &&
         "expect fir.box or fir.class");
  assert(fir::unwrapRefType(info.addr.getType()) == info.boxType &&
         "expected box type consistency");

  if (info.isPresent) {
    llvm::SmallVector<mlir::Type> resTypes;
    constexpr unsigned nbValuesPerBound = 5;
    for (unsigned dim = 0; dim < dataExv.rank() * nbValuesPerBound; ++dim)
      resTypes.push_back(idxTy);

    mlir::Operation::result_range ifRes =
        builder.genIfOp(loc, resTypes, info.isPresent, /*withElseRegion=*/true)
            .genThen([&]() {
              mlir::Value box =
                  !fir::isBoxAddress(info.addr.getType())
                      ? info.addr
                      : builder.create<fir::LoadOp>(loc, info.addr);
              llvm::SmallVector<mlir::Value> boundValues =
                  gatherBoundsOrBoundValues<BoundsOp, BoundsType>(
                      builder, loc, dataExv, box,
                      /*collectValuesOnly=*/true);
              builder.create<fir::ResultOp>(loc, boundValues);
            })
            .genElse([&] {
              // Box is not present. Populate bound values with default values.
              llvm::SmallVector<mlir::Value> boundValues;
              mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
              mlir::Value mOne = builder.createMinusOneInteger(loc, idxTy);
              for (unsigned dim = 0; dim < dataExv.rank(); ++dim) {
                boundValues.push_back(zero); // lb
                boundValues.push_back(mOne); // ub
                boundValues.push_back(zero); // extent
                boundValues.push_back(zero); // byteStride
                boundValues.push_back(zero); // baseLb
              }
              builder.create<fir::ResultOp>(loc, boundValues);
            })
            .getResults();
    // Create the bound operations outside the if-then-else with the if op
    // results.
    for (unsigned i = 0; i < ifRes.size(); i += nbValuesPerBound) {
      mlir::Value bound = builder.create<BoundsOp>(
          loc, boundTy, ifRes[i], ifRes[i + 1], ifRes[i + 2], ifRes[i + 3],
          true, ifRes[i + 4]);
      bounds.push_back(bound);
    }
  } else {
    mlir::Value box = !fir::isBoxAddress(info.addr.getType())
                          ? info.addr
                          : builder.create<fir::LoadOp>(loc, info.addr);
    bounds = gatherBoundsOrBoundValues<BoundsOp, BoundsType>(builder, loc,
                                                             dataExv, box);
  }
  return bounds;
}

/// Generate bounds operation for base array without any subscripts
/// provided.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBaseBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
                 fir::ExtendedValue dataExv, bool isAssumedSize) {
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  if (dataExv.rank() == 0)
    return bounds;

  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  const unsigned rank = dataExv.rank();
  for (unsigned dim = 0; dim < rank; ++dim) {
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value ub;
    mlir::Value lb = zero;
    mlir::Value ext = fir::factory::readExtent(builder, loc, dataExv, dim);
    if (isAssumedSize && dim + 1 == rank) {
      ext = zero;
      ub = lb;
    } else {
      // ub = extent - 1
      ub = builder.create<mlir::arith::SubIOp>(loc, ext, one);
    }

    mlir::Value bound =
        builder.create<BoundsOp>(loc, boundTy, lb, ub, ext, one, false, baseLb);
    bounds.push_back(bound);
  }
  return bounds;
}

namespace detail {
template <typename T> //
static T &&AsRvalueRef(T &&t) {
  return std::move(t);
}
template <typename T> //
static T AsRvalueRef(T &t) {
  return t;
}
template <typename T> //
static T AsRvalueRef(const T &t) {
  return t;
}

// Helper class for stripping enclosing parentheses and a conversion that
// preserves type category. This is used for triplet elements, which are
// always of type integer(kind=8). The lower/upper bounds are converted to
// an "index" type, which is 64-bit, so the explicit conversion to kind=8
// (if present) is not needed. When it's present, though, it causes generated
// names to contain "int(..., kind=8)".
struct PeelConvert {
  template <Fortran::common::TypeCategory Category, int Kind>
  static Fortran::semantics::MaybeExpr visit_with_category(
      const Fortran::evaluate::Expr<Fortran::evaluate::Type<Category, Kind>>
          &expr) {
    return Fortran::common::visit(
        [](auto &&s) { return visit_with_category<Category, Kind>(s); },
        expr.u);
  }
  template <Fortran::common::TypeCategory Category, int Kind>
  static Fortran::semantics::MaybeExpr visit_with_category(
      const Fortran::evaluate::Convert<Fortran::evaluate::Type<Category, Kind>,
                                       Category> &expr) {
    return AsGenericExpr(AsRvalueRef(expr.left()));
  }
  template <Fortran::common::TypeCategory Category, int Kind, typename T>
  static Fortran::semantics::MaybeExpr visit_with_category(const T &) {
    return std::nullopt; //
  }
  template <Fortran::common::TypeCategory Category, typename T>
  static Fortran::semantics::MaybeExpr visit_with_category(const T &) {
    return std::nullopt; //
  }

  template <Fortran::common::TypeCategory Category>
  static Fortran::semantics::MaybeExpr
  visit(const Fortran::evaluate::Expr<Fortran::evaluate::SomeKind<Category>>
            &expr) {
    return Fortran::common::visit(
        [](auto &&s) { return visit_with_category<Category>(s); }, expr.u);
  }
  static Fortran::semantics::MaybeExpr
  visit(const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    return Fortran::common::visit([](auto &&s) { return visit(s); }, expr.u);
  }
  template <typename T> //
  static Fortran::semantics::MaybeExpr visit(const T &) {
    return std::nullopt;
  }
};

static inline Fortran::semantics::SomeExpr
peelOuterConvert(Fortran::semantics::SomeExpr &expr) {
  if (auto peeled = PeelConvert::visit(expr))
    return *peeled;
  return expr;
}
} // namespace detail

/// Generate bounds operations for an array section when subscripts are
/// provided.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::AbstractConverter &converter,
             Fortran::lower::StatementContext &stmtCtx,
             const std::vector<Fortran::evaluate::Subscript> &subscripts,
             std::stringstream &asFortran, fir::ExtendedValue &dataExv,
             bool dataExvIsAssumedSize, AddrAndBoundsInfo &info,
             bool treatIndexAsSection = false) {
  int dimension = 0;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  const int dataExvRank = static_cast<int>(dataExv.rank());
  for (const auto &subscript : subscripts) {
    const auto *triplet{std::get_if<Fortran::evaluate::Triplet>(&subscript.u)};
    if (triplet || treatIndexAsSection) {
      if (dimension != 0)
        asFortran << ',';
      mlir::Value lbound, ubound, extent;
      std::optional<std::int64_t> lval, uval;
      mlir::Value baseLb =
          fir::factory::readLowerBound(builder, loc, dataExv, dimension, one);
      bool defaultLb = baseLb == one;
      mlir::Value stride = one;
      bool strideInBytes = false;

      if (mlir::isa<fir::BaseBoxType>(
              fir::unwrapRefType(info.addr.getType()))) {
        if (info.isPresent) {
          stride =
              builder
                  .genIfOp(loc, idxTy, info.isPresent, /*withElseRegion=*/true)
                  .genThen([&]() {
                    mlir::Value box =
                        !fir::isBoxAddress(info.addr.getType())
                            ? info.addr
                            : builder.create<fir::LoadOp>(loc, info.addr);
                    mlir::Value d =
                        builder.createIntegerConstant(loc, idxTy, dimension);
                    auto dimInfo = builder.create<fir::BoxDimsOp>(
                        loc, idxTy, idxTy, idxTy, box, d);
                    builder.create<fir::ResultOp>(loc, dimInfo.getByteStride());
                  })
                  .genElse([&] {
                    mlir::Value zero =
                        builder.createIntegerConstant(loc, idxTy, 0);
                    builder.create<fir::ResultOp>(loc, zero);
                  })
                  .getResults()[0];
        } else {
          mlir::Value box = !fir::isBoxAddress(info.addr.getType())
                                ? info.addr
                                : builder.create<fir::LoadOp>(loc, info.addr);
          mlir::Value d = builder.createIntegerConstant(loc, idxTy, dimension);
          auto dimInfo =
              builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, d);
          stride = dimInfo.getByteStride();
        }
        strideInBytes = true;
      }

      Fortran::semantics::MaybeExpr lower;
      if (triplet) {
        lower = Fortran::evaluate::AsGenericExpr(triplet->lower());
      } else {
        // Case of IndirectSubscriptIntegerExpr
        using IndirectSubscriptIntegerExpr =
            Fortran::evaluate::IndirectSubscriptIntegerExpr;
        using SubscriptInteger = Fortran::evaluate::SubscriptInteger;
        Fortran::evaluate::Expr<SubscriptInteger> oneInt =
            std::get<IndirectSubscriptIntegerExpr>(subscript.u).value();
        lower = Fortran::evaluate::AsGenericExpr(std::move(oneInt));
        if (lower->Rank() > 0) {
          mlir::emitError(
              loc, "vector subscript cannot be used for an array section");
          break;
        }
      }
      if (lower) {
        lval = Fortran::evaluate::ToInt64(*lower);
        if (lval) {
          if (defaultLb) {
            lbound = builder.createIntegerConstant(loc, idxTy, *lval - 1);
          } else {
            mlir::Value lb = builder.createIntegerConstant(loc, idxTy, *lval);
            lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          }
          asFortran << *lval;
        } else {
          mlir::Value lb =
              fir::getBase(converter.genExprValue(loc, *lower, stmtCtx));
          lb = builder.createConvert(loc, baseLb.getType(), lb);
          lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          asFortran << detail::peelOuterConvert(*lower).AsFortran();
        }
      } else {
        // If the lower bound is not specified, then the section
        // starts from offset 0 of the dimension.
        // Note that the lowerbound in the BoundsOp is always 0-based.
        lbound = zero;
      }

      if (!triplet) {
        // If it is a scalar subscript, then the upper bound
        // is equal to the lower bound, and the extent is one.
        ubound = lbound;
        extent = one;
      } else {
        asFortran << ':';
        Fortran::semantics::MaybeExpr upper =
            Fortran::evaluate::AsGenericExpr(triplet->upper());

        if (upper) {
          uval = Fortran::evaluate::ToInt64(*upper);
          if (uval) {
            if (defaultLb) {
              ubound = builder.createIntegerConstant(loc, idxTy, *uval - 1);
            } else {
              mlir::Value ub = builder.createIntegerConstant(loc, idxTy, *uval);
              ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
            }
            asFortran << *uval;
          } else {
            mlir::Value ub =
                fir::getBase(converter.genExprValue(loc, *upper, stmtCtx));
            ub = builder.createConvert(loc, baseLb.getType(), ub);
            ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
            asFortran << detail::peelOuterConvert(*upper).AsFortran();
          }
        }
        if (lower && upper) {
          if (lval && uval && *uval < *lval) {
            mlir::emitError(loc, "zero sized array section");
            break;
          } else {
            // Stride is mandatory in evaluate::Triplet. Make sure it's 1.
            auto val = Fortran::evaluate::ToInt64(triplet->GetStride());
            if (!val || *val != 1) {
              mlir::emitError(loc, "stride cannot be specified on "
                                   "an array section");
              break;
            }
          }
        }

        if (info.isPresent && mlir::isa<fir::BaseBoxType>(
                                  fir::unwrapRefType(info.addr.getType()))) {
          extent =
              builder
                  .genIfOp(loc, idxTy, info.isPresent, /*withElseRegion=*/true)
                  .genThen([&]() {
                    mlir::Value ext = fir::factory::readExtent(
                        builder, loc, dataExv, dimension);
                    builder.create<fir::ResultOp>(loc, ext);
                  })
                  .genElse([&] {
                    mlir::Value zero =
                        builder.createIntegerConstant(loc, idxTy, 0);
                    builder.create<fir::ResultOp>(loc, zero);
                  })
                  .getResults()[0];
        } else {
          extent = fir::factory::readExtent(builder, loc, dataExv, dimension);
        }

        if (dataExvIsAssumedSize && dimension + 1 == dataExvRank) {
          extent = zero;
          if (ubound && lbound) {
            mlir::Value diff =
                builder.create<mlir::arith::SubIOp>(loc, ubound, lbound);
            extent = builder.create<mlir::arith::AddIOp>(loc, diff, one);
          }
          if (!ubound)
            ubound = lbound;
        }

        if (!ubound) {
          // ub = extent - 1
          ubound = builder.create<mlir::arith::SubIOp>(loc, extent, one);
        }
      }
      mlir::Value bound = builder.create<BoundsOp>(
          loc, boundTy, lbound, ubound, extent, stride, strideInBytes, baseLb);
      bounds.push_back(bound);
      ++dimension;
    }
  }
  return bounds;
}

namespace detail {
template <typename Ref, typename Expr> //
std::optional<Ref> getRef(Expr &&expr) {
  if constexpr (std::is_same_v<llvm::remove_cvref_t<Expr>,
                               Fortran::evaluate::DataRef>) {
    if (auto *ref = std::get_if<Ref>(&expr.u))
      return *ref;
    return std::nullopt;
  } else {
    auto maybeRef = Fortran::evaluate::ExtractDataRef(expr);
    if (!maybeRef || !std::holds_alternative<Ref>(maybeRef->u))
      return std::nullopt;
    return std::get<Ref>(maybeRef->u);
  }
}
} // namespace detail

template <typename BoundsOp, typename BoundsType>
AddrAndBoundsInfo gatherDataOperandAddrAndBounds(
    Fortran::lower::AbstractConverter &converter, fir::FirOpBuilder &builder,
    semantics::SemanticsContext &semaCtx,
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::semantics::SymbolRef symbol,
    const Fortran::semantics::MaybeExpr &maybeDesignator,
    mlir::Location operandLocation, std::stringstream &asFortran,
    llvm::SmallVector<mlir::Value> &bounds, bool treatIndexAsSection = false) {
  using namespace Fortran;

  AddrAndBoundsInfo info;

  if (!maybeDesignator) {
    info = getDataOperandBaseAddr(converter, builder, symbol, operandLocation);
    asFortran << symbol->name().ToString();
    return info;
  }

  semantics::SomeExpr designator = *maybeDesignator;

  if ((designator.Rank() > 0 || treatIndexAsSection) &&
      IsArrayElement(designator)) {
    auto arrayRef = detail::getRef<evaluate::ArrayRef>(designator);
    // This shouldn't fail after IsArrayElement(designator).
    assert(arrayRef && "Expecting ArrayRef");

    fir::ExtendedValue dataExv;
    bool dataExvIsAssumedSize = false;

    auto toMaybeExpr = [&](auto &&base) {
      using BaseType = llvm::remove_cvref_t<decltype(base)>;
      evaluate::ExpressionAnalyzer ea{semaCtx};

      if constexpr (std::is_same_v<evaluate::NamedEntity, BaseType>) {
        if (auto *ref = base.UnwrapSymbolRef())
          return ea.Designate(evaluate::DataRef{*ref});
        if (auto *ref = base.UnwrapComponent())
          return ea.Designate(evaluate::DataRef{*ref});
        llvm_unreachable("Unexpected NamedEntity");
      } else {
        static_assert(std::is_same_v<semantics::SymbolRef, BaseType>);
        return ea.Designate(evaluate::DataRef{base});
      }
    };

    auto arrayBase = toMaybeExpr(arrayRef->base());
    assert(arrayBase);

    if (detail::getRef<evaluate::Component>(*arrayBase)) {
      dataExv = converter.genExprAddr(operandLocation, *arrayBase, stmtCtx);
      info.addr = fir::getBase(dataExv);
      info.rawInput = info.addr;
      asFortran << arrayBase->AsFortran();
    } else {
      const semantics::Symbol &sym = arrayRef->GetLastSymbol();
      dataExvIsAssumedSize =
          Fortran::semantics::IsAssumedSizeArray(sym.GetUltimate());
      info = getDataOperandBaseAddr(converter, builder, sym, operandLocation);
      dataExv = converter.getSymbolExtendedValue(sym);
      asFortran << sym.name().ToString();
    }

    if (!arrayRef->subscript().empty()) {
      asFortran << '(';
      bounds = genBoundsOps<BoundsOp, BoundsType>(
          builder, operandLocation, converter, stmtCtx, arrayRef->subscript(),
          asFortran, dataExv, dataExvIsAssumedSize, info, treatIndexAsSection);
    }
    asFortran << ')';
  } else if (auto compRef = detail::getRef<evaluate::Component>(designator)) {
    fir::ExtendedValue compExv =
        converter.genExprAddr(operandLocation, designator, stmtCtx);
    info.addr = fir::getBase(compExv);
    info.rawInput = info.addr;
    if (mlir::isa<fir::SequenceType>(fir::unwrapRefType(info.addr.getType())))
      bounds = genBaseBoundsOps<BoundsOp, BoundsType>(builder, operandLocation,
                                                      compExv,
                                                      /*isAssumedSize=*/false);
    asFortran << designator.AsFortran();

    if (semantics::IsOptional(compRef->GetLastSymbol())) {
      info.isPresent = builder.create<fir::IsPresentOp>(
          operandLocation, builder.getI1Type(), info.rawInput);
    }

    if (auto loadOp =
            mlir::dyn_cast_or_null<fir::LoadOp>(info.addr.getDefiningOp())) {
      if (fir::isAllocatableType(loadOp.getType()) ||
          fir::isPointerType(loadOp.getType())) {
        info.boxType = info.addr.getType();
        info.addr = builder.create<fir::BoxAddrOp>(operandLocation, info.addr);
      }
      info.rawInput = info.addr;
    }

    // If the component is an allocatable or pointer the result of
    // genExprAddr will be the result of a fir.box_addr operation or
    // a fir.box_addr has been inserted just before.
    // Retrieve the box so we handle it like other descriptor.
    if (auto boxAddrOp =
            mlir::dyn_cast_or_null<fir::BoxAddrOp>(info.addr.getDefiningOp())) {
      info.addr = boxAddrOp.getVal();
      info.boxType = info.addr.getType();
      info.rawInput = info.addr;
      bounds = genBoundsOpsFromBox<BoundsOp, BoundsType>(
          builder, operandLocation, compExv, info);
    }
  } else {
    if (detail::getRef<evaluate::ArrayRef>(designator)) {
      fir::ExtendedValue compExv =
          converter.genExprAddr(operandLocation, designator, stmtCtx);
      info.addr = fir::getBase(compExv);
      info.rawInput = info.addr;
      asFortran << designator.AsFortran();
    } else if (auto symRef = detail::getRef<semantics::SymbolRef>(designator)) {
      // Scalar or full array.
      fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(*symRef);
      info =
          getDataOperandBaseAddr(converter, builder, *symRef, operandLocation);
      if (mlir::isa<fir::BaseBoxType>(
              fir::unwrapRefType(info.addr.getType()))) {
        info.boxType = fir::unwrapRefType(info.addr.getType());
        bounds = genBoundsOpsFromBox<BoundsOp, BoundsType>(
            builder, operandLocation, dataExv, info);
      }
      bool dataExvIsAssumedSize =
          Fortran::semantics::IsAssumedSizeArray(symRef->get().GetUltimate());
      if (mlir::isa<fir::SequenceType>(fir::unwrapRefType(info.addr.getType())))
        bounds = genBaseBoundsOps<BoundsOp, BoundsType>(
            builder, operandLocation, dataExv, dataExvIsAssumedSize);
      asFortran << symRef->get().name().ToString();
    } else { // Unsupported
      llvm::report_fatal_error("Unsupported type of OpenACC operand");
    }
  }

  return info;
}
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DIRECTIVES_COMMON_H
