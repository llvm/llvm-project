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
#include "flang/Optimizer/Builder/Todo.h"
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
  explicit AddrAndBoundsInfo(mlir::Value addr) : addr(addr) {}
  explicit AddrAndBoundsInfo(mlir::Value addr, mlir::Value isPresent)
      : addr(addr), isPresent(isPresent) {}
  mlir::Value addr = nullptr;
  mlir::Value isPresent = nullptr;
};

/// Checks if the assignment statement has a single variable on the RHS.
static inline bool checkForSingleVariableOnRHS(
    const Fortran::parser::AssignmentStmt &assignmentStmt) {
  const Fortran::parser::Expr &expr{
      std::get<Fortran::parser::Expr>(assignmentStmt.t)};
  const Fortran::common::Indirection<Fortran::parser::Designator> *designator =
      std::get_if<Fortran::common::Indirection<Fortran::parser::Designator>>(
          &expr.u);
  return designator != nullptr;
}

/// Checks if the symbol on the LHS of the assignment statement is present in
/// the RHS expression.
static inline bool
checkForSymbolMatch(const Fortran::parser::AssignmentStmt &assignmentStmt) {
  const auto &var{std::get<Fortran::parser::Variable>(assignmentStmt.t)};
  const auto &expr{std::get<Fortran::parser::Expr>(assignmentStmt.t)};
  const auto *e{Fortran::semantics::GetExpr(expr)};
  const auto *v{Fortran::semantics::GetExpr(var)};
  auto varSyms{Fortran::evaluate::GetSymbolVector(*v)};
  const Fortran::semantics::Symbol &varSymbol{*varSyms.front()};
  for (const Fortran::semantics::Symbol &symbol :
       Fortran::evaluate::GetSymbolVector(*e))
    if (varSymbol == symbol)
      return true;
  return false;
}

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
  if (checkForSingleVariableOnRHS(stmt1)) {
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
  if (checkForSingleVariableOnRHS(stmt1)) {
    if (checkForSymbolMatch(stmt2)) {
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
  // TODO: Might need revisiting to handle for non-shared clauses
  if (!symAddr) {
    if (const auto *details =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>())
      symAddr = converter.getSymbolAddress(details->symbol());
  }

  if (!symAddr)
    llvm::report_fatal_error("could not retrieve symbol address");

  if (auto boxTy =
          fir::unwrapRefType(symAddr.getType()).dyn_cast<fir::BaseBoxType>()) {
    if (boxTy.getEleTy().isa<fir::RecordType>())
      TODO(loc, "derived type");

    // Load the box when baseAddr is a `fir.ref<fir.box<T>>` or a
    // `fir.ref<fir.class<T>>` type.
    if (symAddr.getType().isa<fir::ReferenceType>()) {
      if (Fortran::semantics::IsOptional(sym)) {
        mlir::Value isPresent =
            builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), symAddr);
        mlir::Value addr =
            builder.genIfOp(loc, {boxTy}, isPresent, /*withElseRegion=*/true)
                .genThen([&]() {
                  mlir::Value load = builder.create<fir::LoadOp>(loc, symAddr);
                  builder.create<fir::ResultOp>(loc, mlir::ValueRange{load});
                })
                .genElse([&] {
                  mlir::Value absent =
                      builder.create<fir::AbsentOp>(loc, boxTy);
                  builder.create<fir::ResultOp>(loc, mlir::ValueRange{absent});
                })
                .getResults()[0];
        return AddrAndBoundsInfo(addr, isPresent);
      }
      mlir::Value addr = builder.create<fir::LoadOp>(loc, symAddr);
      return AddrAndBoundsInfo(addr);
      ;
    }
  }
  return AddrAndBoundsInfo(symAddr);
}

template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
gatherBoundsOrBoundValues(fir::FirOpBuilder &builder, mlir::Location loc,
                          fir::ExtendedValue dataExv, mlir::Value box,
                          bool collectValuesOnly = false) {
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
                    Fortran::lower::AbstractConverter &converter,
                    fir::ExtendedValue dataExv,
                    Fortran::lower::AddrAndBoundsInfo &info) {
  llvm::SmallVector<mlir::Value> bounds;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();

  assert(info.addr.getType().isa<fir::BaseBoxType>() &&
         "expect fir.box or fir.class");

  if (info.isPresent) {
    llvm::SmallVector<mlir::Type> resTypes;
    constexpr unsigned nbValuesPerBound = 5;
    for (unsigned dim = 0; dim < dataExv.rank() * nbValuesPerBound; ++dim)
      resTypes.push_back(idxTy);

    mlir::Operation::result_range ifRes =
        builder.genIfOp(loc, resTypes, info.isPresent, /*withElseRegion=*/true)
            .genThen([&]() {
              llvm::SmallVector<mlir::Value> boundValues =
                  gatherBoundsOrBoundValues<BoundsOp, BoundsType>(
                      builder, loc, dataExv, info.addr,
                      /*collectValuesOnly=*/true);
              builder.create<fir::ResultOp>(loc, boundValues);
            })
            .genElse([&] {
              // Box is not present. Populate bound values with default values.
              llvm::SmallVector<mlir::Value> boundValues;
              mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
              mlir::Value mOne = builder.createIntegerConstant(loc, idxTy, -1);
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
    bounds = gatherBoundsOrBoundValues<BoundsOp, BoundsType>(
        builder, loc, dataExv, info.addr);
  }
  return bounds;
}

/// Generate bounds operation for base array without any subscripts
/// provided.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBaseBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
                 Fortran::lower::AbstractConverter &converter,
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

/// Generate bounds operations for an array section when subscripts are
/// provided.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::AbstractConverter &converter,
             Fortran::lower::StatementContext &stmtCtx,
             const std::list<Fortran::parser::SectionSubscript> &subscripts,
             std::stringstream &asFortran, fir::ExtendedValue &dataExv,
             bool dataExvIsAssumedSize, mlir::Value baseAddr,
             bool treatIndexAsSection = false) {
  int dimension = 0;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  const int dataExvRank = static_cast<int>(dataExv.rank());
  for (const auto &subscript : subscripts) {
    const auto *triplet{
        std::get_if<Fortran::parser::SubscriptTriplet>(&subscript.u)};
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

      if (fir::unwrapRefType(baseAddr.getType()).isa<fir::BaseBoxType>()) {
        mlir::Value d = builder.createIntegerConstant(loc, idxTy, dimension);
        auto dimInfo = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                      baseAddr, d);
        stride = dimInfo.getByteStride();
        strideInBytes = true;
      }

      const Fortran::lower::SomeExpr *lower{nullptr};
      if (triplet) {
        if (const auto &tripletLb{std::get<0>(triplet->t)})
          lower = Fortran::semantics::GetExpr(*tripletLb);
      } else {
        const auto &index{std::get<Fortran::parser::IntExpr>(subscript.u)};
        lower = Fortran::semantics::GetExpr(index);
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
          asFortran << lower->AsFortran();
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
        const auto &upper{std::get<1>(triplet->t)};

        if (upper) {
          uval = Fortran::semantics::GetIntValue(upper);
          if (uval) {
            if (defaultLb) {
              ubound = builder.createIntegerConstant(loc, idxTy, *uval - 1);
            } else {
              mlir::Value ub = builder.createIntegerConstant(loc, idxTy, *uval);
              ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
            }
            asFortran << *uval;
          } else {
            const Fortran::lower::SomeExpr *uexpr =
                Fortran::semantics::GetExpr(*upper);
            mlir::Value ub =
                fir::getBase(converter.genExprValue(loc, *uexpr, stmtCtx));
            ub = builder.createConvert(loc, baseLb.getType(), ub);
            ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
            asFortran << uexpr->AsFortran();
          }
        }
        if (lower && upper) {
          if (lval && uval && *uval < *lval) {
            mlir::emitError(loc, "zero sized array section");
            break;
          } else if (std::get<2>(triplet->t)) {
            const auto &strideExpr{std::get<2>(triplet->t)};
            if (strideExpr) {
              mlir::emitError(loc, "stride cannot be specified on "
                                   "an array section");
              break;
            }
          }
        }

        extent = fir::factory::readExtent(builder, loc, dataExv, dimension);
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

template <typename ObjectType, typename BoundsOp, typename BoundsType>
AddrAndBoundsInfo gatherDataOperandAddrAndBounds(
    Fortran::lower::AbstractConverter &converter, fir::FirOpBuilder &builder,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx, const ObjectType &object,
    mlir::Location operandLocation, std::stringstream &asFortran,
    llvm::SmallVector<mlir::Value> &bounds, bool treatIndexAsSection = false) {
  AddrAndBoundsInfo info;
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto expr{Fortran::semantics::AnalyzeExpr(semanticsContext,
                                                          designator)}) {
              if (((*expr).Rank() > 0 || treatIndexAsSection) &&
                  Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                      designator)) {
                const auto *arrayElement =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator);
                const auto *dataRef =
                    std::get_if<Fortran::parser::DataRef>(&designator.u);
                fir::ExtendedValue dataExv;
                bool dataExvIsAssumedSize = false;
                if (Fortran::parser::Unwrap<
                        Fortran::parser::StructureComponent>(
                        arrayElement->base)) {
                  auto exprBase = Fortran::semantics::AnalyzeExpr(
                      semanticsContext, arrayElement->base);
                  dataExv = converter.genExprAddr(operandLocation, *exprBase,
                                                  stmtCtx);
                  info.addr = fir::getBase(dataExv);
                  asFortran << (*exprBase).AsFortran();
                } else {
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  dataExvIsAssumedSize = Fortran::semantics::IsAssumedSizeArray(
                      name.symbol->GetUltimate());
                  info = getDataOperandBaseAddr(converter, builder,
                                                *name.symbol, operandLocation);
                  dataExv = converter.getSymbolExtendedValue(*name.symbol);
                  asFortran << name.ToString();
                }

                if (!arrayElement->subscripts.empty()) {
                  asFortran << '(';
                  bounds = genBoundsOps<BoundsOp, BoundsType>(
                      builder, operandLocation, converter, stmtCtx,
                      arrayElement->subscripts, asFortran, dataExv,
                      dataExvIsAssumedSize, info.addr, treatIndexAsSection);
                }
                asFortran << ')';
              } else if (auto structComp = Fortran::parser::Unwrap<
                             Fortran::parser::StructureComponent>(designator)) {
                fir::ExtendedValue compExv =
                    converter.genExprAddr(operandLocation, *expr, stmtCtx);
                info.addr = fir::getBase(compExv);
                if (fir::unwrapRefType(info.addr.getType())
                        .isa<fir::SequenceType>())
                  bounds = genBaseBoundsOps<BoundsOp, BoundsType>(
                      builder, operandLocation, converter, compExv,
                      /*isAssumedSize=*/false);
                asFortran << (*expr).AsFortran();

                bool isOptional = Fortran::semantics::IsOptional(
                    *Fortran::parser::GetLastName(*structComp).symbol);
                if (isOptional)
                  info.isPresent = builder.create<fir::IsPresentOp>(
                      operandLocation, builder.getI1Type(), info.addr);

                if (auto loadOp = mlir::dyn_cast_or_null<fir::LoadOp>(
                        info.addr.getDefiningOp())) {
                  if (fir::isAllocatableType(loadOp.getType()) ||
                      fir::isPointerType(loadOp.getType()))
                    info.addr = builder.create<fir::BoxAddrOp>(operandLocation,
                                                               info.addr);
                }

                // If the component is an allocatable or pointer the result of
                // genExprAddr will be the result of a fir.box_addr operation or
                // a fir.box_addr has been inserted just before.
                // Retrieve the box so we handle it like other descriptor.
                if (auto boxAddrOp = mlir::dyn_cast_or_null<fir::BoxAddrOp>(
                        info.addr.getDefiningOp())) {
                  info.addr = boxAddrOp.getVal();
                  bounds = genBoundsOpsFromBox<BoundsOp, BoundsType>(
                      builder, operandLocation, converter, compExv, info);
                }
              } else {
                if (Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator)) {
                  // Single array element.
                  const auto *arrayElement =
                      Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                          designator);
                  (void)arrayElement;
                  fir::ExtendedValue compExv =
                      converter.genExprAddr(operandLocation, *expr, stmtCtx);
                  info.addr = fir::getBase(compExv);
                  asFortran << (*expr).AsFortran();
                } else if (const auto *dataRef{
                               std::get_if<Fortran::parser::DataRef>(
                                   &designator.u)}) {
                  // Scalar or full array.
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  fir::ExtendedValue dataExv =
                      converter.getSymbolExtendedValue(*name.symbol);
                  info = getDataOperandBaseAddr(converter, builder,
                                                *name.symbol, operandLocation);
                  if (fir::unwrapRefType(info.addr.getType())
                          .isa<fir::BaseBoxType>()) {
                    bounds = genBoundsOpsFromBox<BoundsOp, BoundsType>(
                        builder, operandLocation, converter, dataExv, info);
                  }
                  bool dataExvIsAssumedSize =
                      Fortran::semantics::IsAssumedSizeArray(
                          name.symbol->GetUltimate());
                  if (fir::unwrapRefType(info.addr.getType())
                          .isa<fir::SequenceType>())
                    bounds = genBaseBoundsOps<BoundsOp, BoundsType>(
                        builder, operandLocation, converter, dataExv,
                        dataExvIsAssumedSize);
                  asFortran << name.ToString();
                } else { // Unsupported
                  llvm::report_fatal_error(
                      "Unsupported type of OpenACC operand");
                }
              }
            }
          },
          [&](const Fortran::parser::Name &name) {
            info = getDataOperandBaseAddr(converter, builder, *name.symbol,
                                          operandLocation);
            asFortran << name.ToString();
          }},
      object.u);
  return info;
}

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DIRECTIVES_COMMON_H
