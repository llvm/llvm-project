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
/// be used for both declarations and templated/inline implementations.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_DIRECTIVES_COMMON_H
#define FORTRAN_LOWER_DIRECTIVES_COMMON_H

#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
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
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <type_traits>

namespace Fortran {
namespace lower {

/// Checks if the assignment statement has a single variable on the RHS.
static inline bool checkForSingleVariableOnRHS(
    const Fortran::parser::AssignmentStmt &assignmentStmt) {
  const Fortran::parser::Expr &expr{
      std::get<Fortran::parser::Expr>(assignmentStmt.t)};
  const Fortran::common::Indirection<Fortran::parser::Designator> *designator =
      std::get_if<Fortran::common::Indirection<Fortran::parser::Designator>>(
          &expr.u);
  const Fortran::parser::Name *name =
      designator
          ? Fortran::semantics::getDesignatorNameIfDataRef(designator->value())
          : nullptr;
  return name != nullptr;
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
    mlir::Type elementType) {
  // Generate `atomic.read` operation for atomic assigment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

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
        currentLocation, fromAddress, toAddress,
        mlir::TypeAttr::get(elementType), hint, memoryOrder);
  } else {
    firOpBuilder.create<mlir::acc::AtomicReadOp>(
        currentLocation, fromAddress, toAddress,
        mlir::TypeAttr::get(elementType));
  }
}

/// Used to generate atomic.write operation which is created in existing
/// location set by builder.
template <typename AtomicListT>
static inline void genOmpAccAtomicWriteStatement(
    Fortran::lower::AbstractConverter &converter, mlir::Value lhsAddr,
    mlir::Value rhsExpr, [[maybe_unused]] const AtomicListT *leftHandClauseList,
    [[maybe_unused]] const AtomicListT *rightHandClauseList,
    mlir::Value *evaluatedExprValue = nullptr) {
  // Generate `atomic.write` operation for atomic assignment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

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
    firOpBuilder.create<mlir::omp::AtomicWriteOp>(currentLocation, lhsAddr,
                                                  rhsExpr, hint, memoryOrder);
  } else {
    firOpBuilder.create<mlir::acc::AtomicWriteOp>(currentLocation, lhsAddr,
                                                  rhsExpr);
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
    [[maybe_unused]] const AtomicListT *rightHandClauseList) {
  // Generate `omp.atomic.update` operation for atomic assignment statements
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const auto *varDesignator =
      std::get_if<Fortran::common::Indirection<Fortran::parser::Designator>>(
          &assignmentStmtVariable.u);
  assert(varDesignator && "Variable designator for atomic update assignment "
                          "statement does not exist");
  const Fortran::parser::Name *name =
      Fortran::semantics::getDesignatorNameIfDataRef(varDesignator->value());
  if (!name)
    TODO(converter.getCurrentLocation(),
         "Array references as atomic update variable");
  assert(name && name->symbol &&
         "No symbol attached to atomic update variable");
  if (Fortran::semantics::IsAllocatableOrPointer(name->symbol->GetUltimate()))
    converter.bindSymbol(*name->symbol, lhsAddr);

  //  Lowering is in two steps :
  //  subroutine sb
  //    integer :: a, b
  //    !$omp atomic update
  //      a = a + b
  //  end subroutine
  //
  //  1. Lower to scf.execute_region_op
  //
  //  func.func @_QPsb() {
  //    %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFsbEa"}
  //    %1 = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFsbEb"}
  //    %2 = scf.execute_region -> i32 {
  //      %3 = fir.load %0 : !fir.ref<i32>
  //      %4 = fir.load %1 : !fir.ref<i32>
  //      %5 = arith.addi %3, %4 : i32
  //      scf.yield %5 : i32
  //    }
  //    return
  //  }
  auto tempOp =
      firOpBuilder.create<mlir::scf::ExecuteRegionOp>(currentLocation, varType);
  firOpBuilder.createBlock(&tempOp.getRegion());
  mlir::Block &block = tempOp.getRegion().back();
  firOpBuilder.setInsertionPointToEnd(&block);
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value rhsExpr = fir::getBase(converter.genExprValue(
      *Fortran::semantics::GetExpr(assignmentStmtExpr), stmtCtx));
  mlir::Value convertResult =
      firOpBuilder.createConvert(currentLocation, varType, rhsExpr);
  // Insert the terminator: YieldOp.
  firOpBuilder.create<mlir::scf::YieldOp>(currentLocation, convertResult);
  firOpBuilder.setInsertionPointToStart(&block);

  //  2. Create the omp.atomic.update Operation using the Operations in the
  //     temporary scf.execute_region Operation.
  //
  //  func.func @_QPsb() {
  //    %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFsbEa"}
  //    %1 = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFsbEb"}
  //    %2 = fir.load %1 : !fir.ref<i32>
  //    omp.atomic.update   %0 : !fir.ref<i32> {
  //    ^bb0(%arg0: i32):
  //      %3 = fir.load %1 : !fir.ref<i32>
  //      %4 = arith.addi %arg0, %3 : i32
  //      omp.yield(%3 : i32)
  //    }
  //    return
  //  }
  mlir::Value updateVar = converter.getSymbolAddress(*name->symbol);
  if (auto decl = updateVar.getDefiningOp<hlfir::DeclareOp>())
    updateVar = decl.getBase();

  firOpBuilder.setInsertionPointAfter(tempOp);

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
        currentLocation, updateVar, hint, memoryOrder);
  } else {
    atomicUpdateOp = firOpBuilder.create<mlir::acc::AtomicUpdateOp>(
        currentLocation, updateVar);
  }

  llvm::SmallVector<mlir::Type> varTys = {varType};
  llvm::SmallVector<mlir::Location> locs = {currentLocation};
  firOpBuilder.createBlock(&atomicUpdateOp->getRegion(0), {}, varTys, locs);
  mlir::Value val =
      fir::getBase(atomicUpdateOp->getRegion(0).front().getArgument(0));

  llvm::SmallVector<mlir::Operation *> ops;
  for (mlir::Operation &op : tempOp.getRegion().getOps())
    ops.push_back(&op);

  // SCF Yield is converted to OMP Yield. All other operations are copied
  for (mlir::Operation *op : ops) {
    if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      firOpBuilder.setInsertionPointToEnd(
          &atomicUpdateOp->getRegion(0).front());
      if constexpr (std::is_same<AtomicListT,
                                 Fortran::parser::OmpAtomicClauseList>()) {
        firOpBuilder.create<mlir::omp::YieldOp>(currentLocation,
                                                y.getResults());
      } else {
        firOpBuilder.create<mlir::acc::YieldOp>(currentLocation,
                                                y.getResults());
      }
      op->erase();
    } else {
      op->remove();
      atomicUpdateOp->getRegion(0).front().push_back(op);
    }
  }

  // Remove the load and replace all uses of load with the block argument
  for (mlir::Operation &op : atomicUpdateOp->getRegion(0).getOps()) {
    fir::LoadOp y = mlir::dyn_cast<fir::LoadOp>(&op);
    if (y && y.getMemref() == updateVar)
      y.getRes().replaceAllUsesWith(val);
  }

  tempOp.erase();
}

/// Processes an atomic construct with write clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicWrite(Fortran::lower::AbstractConverter &converter,
                          const AtomicT &atomicWrite) {
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
                                rightHandClauseList);
}

/// Processes an atomic construct with read clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicRead(Fortran::lower::AbstractConverter &converter,
                         const AtomicT &atomicRead) {
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
  genOmpAccAtomicCaptureStatement(converter, fromAddress, toAddress,
                                  leftHandClauseList, rightHandClauseList,
                                  elementType);
}

/// Processes an atomic construct with update clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicUpdate(Fortran::lower::AbstractConverter &converter,
                           const AtomicT &atomicUpdate) {
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
  mlir::Type varType =
      fir::getBase(
          converter.genExprValue(
              *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx))
          .getType();
  genOmpAccAtomicUpdateStatement<AtomicListT>(
      converter, lhsAddr, varType, assignmentStmtVariable, assignmentStmtExpr,
      leftHandClauseList, rightHandClauseList);
}

/// Processes an atomic construct with no clause - which implies update clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAtomic(Fortran::lower::AbstractConverter &converter,
                  const AtomicT &atomicConstruct) {
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
  mlir::Type varType =
      fir::getBase(
          converter.genExprValue(
              *Fortran::semantics::GetExpr(assignmentStmtVariable), stmtCtx))
          .getType();
  // If atomic-clause is not present on the construct, the behaviour is as if
  // the update clause is specified (for both OpenMP and OpenACC).
  genOmpAccAtomicUpdateStatement<AtomicListT>(
      converter, lhsAddr, varType, assignmentStmtVariable, assignmentStmtExpr,
      &atomicClauseList, nullptr);
}

/// Processes an atomic construct with capture clause.
template <typename AtomicT, typename AtomicListT>
void genOmpAccAtomicCapture(Fortran::lower::AbstractConverter &converter,
                            const AtomicT &atomicCapture) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const Fortran::parser::AssignmentStmt &stmt1 =
      std::get<typename AtomicT::Stmt1>(atomicCapture.t).v.statement;
  const auto &stmt1Var{std::get<Fortran::parser::Variable>(stmt1.t)};
  const auto &stmt1Expr{std::get<Fortran::parser::Expr>(stmt1.t)};
  const Fortran::parser::AssignmentStmt &stmt2 =
      std::get<typename AtomicT::Stmt2>(atomicCapture.t).v.statement;
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
  stmt1LHSArg = fir::getBase(
      converter.genExprAddr(*Fortran::semantics::GetExpr(stmt1Var), stmtCtx));
  stmt2LHSArg = fir::getBase(
      converter.genExprAddr(*Fortran::semantics::GetExpr(stmt2Var), stmtCtx));

  // Operation specific RHS evaluations
  if (checkForSingleVariableOnRHS(stmt1)) {
    // Atomic capture construct is of the form [capture-stmt, update-stmt] or
    // of the form [capture-stmt, write-stmt]
    stmt1RHSArg = fir::getBase(converter.genExprAddr(
        *Fortran::semantics::GetExpr(stmt1Expr), stmtCtx));
    stmt2RHSArg = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(stmt2Expr), stmtCtx));

  } else {
    // Atomic capture construct is of the form [update-stmt, capture-stmt]
    stmt1RHSArg = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(stmt1Expr), stmtCtx));
    stmt2RHSArg = fir::getBase(converter.genExprAddr(
        *Fortran::semantics::GetExpr(stmt2Expr), stmtCtx));
  }
  // Type information used in generation of `atomic.update` operation
  mlir::Type stmt1VarType =
      fir::getBase(converter.genExprValue(
                       *Fortran::semantics::GetExpr(stmt1Var), stmtCtx))
          .getType();
  mlir::Type stmt2VarType =
      fir::getBase(converter.genExprValue(
                       *Fortran::semantics::GetExpr(stmt2Var), stmtCtx))
          .getType();

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
    atomicCaptureOp = firOpBuilder.create<mlir::omp::AtomicCaptureOp>(
        currentLocation, hint, memoryOrder);
  } else {
    atomicCaptureOp =
        firOpBuilder.create<mlir::acc::AtomicCaptureOp>(currentLocation);
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
          /*rightHandClauseList=*/nullptr, elementType);
      genOmpAccAtomicUpdateStatement<AtomicListT>(
          converter, stmt1RHSArg, stmt2VarType, stmt2Var, stmt2Expr,
          /*leftHandClauseList=*/nullptr,
          /*rightHandClauseList=*/nullptr);
    } else {
      // Atomic capture construct is of the form [capture-stmt, write-stmt]
      const Fortran::semantics::SomeExpr &fromExpr =
          *Fortran::semantics::GetExpr(stmt1Expr);
      elementType = converter.genType(fromExpr);
      genOmpAccAtomicCaptureStatement<AtomicListT>(
          converter, stmt1RHSArg, stmt1LHSArg,
          /*leftHandClauseList=*/nullptr,
          /*rightHandClauseList=*/nullptr, elementType);
      genOmpAccAtomicWriteStatement<AtomicListT>(
          converter, stmt1RHSArg, stmt2RHSArg,
          /*leftHandClauseList=*/nullptr,
          /*rightHandClauseList=*/nullptr);
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
        /*rightHandClauseList=*/nullptr, elementType);
    firOpBuilder.setInsertionPointToStart(&block);
    genOmpAccAtomicUpdateStatement<AtomicListT>(
        converter, stmt1LHSArg, stmt1VarType, stmt1Var, stmt1Expr,
        /*leftHandClauseList=*/nullptr,
        /*rightHandClauseList=*/nullptr);
  }
  firOpBuilder.setInsertionPointToEnd(&block);
  if constexpr (std::is_same<AtomicListT,
                             Fortran::parser::OmpAtomicClauseList>()) {
    firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);
  } else {
    firOpBuilder.create<mlir::acc::TerminatorOp>(currentLocation);
  }
  firOpBuilder.setInsertionPointToStart(&block);
}

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DIRECTIVES_COMMON_H