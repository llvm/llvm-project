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

inline mlir::Value
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
    if (symAddr.getType().isa<fir::ReferenceType>())
      return builder.create<fir::LoadOp>(loc, symAddr);
  }
  return symAddr;
}

/// Generate the bounds operation from the descriptor information.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBoundsOpsFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                    Fortran::lower::AbstractConverter &converter,
                    fir::ExtendedValue dataExv, mlir::Value box) {
  llvm::SmallVector<mlir::Value> bounds;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  assert(box.getType().isa<fir::BaseBoxType>() &&
         "expect fir.box or fir.class");
  for (unsigned dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value d = builder.createIntegerConstant(loc, idxTy, dim);
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    auto dimInfo =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, d);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value ub =
        builder.create<mlir::arith::SubIOp>(loc, dimInfo.getExtent(), one);
    mlir::Value bound =
        builder.create<BoundsOp>(loc, boundTy, lb, ub, mlir::Value(),
                                 dimInfo.getByteStride(), true, baseLb);
    bounds.push_back(bound);
  }
  return bounds;
}

/// Generate bounds operation for base array without any subscripts
/// provided.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBaseBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
                 Fortran::lower::AbstractConverter &converter,
                 fir::ExtendedValue dataExv, mlir::Value baseAddr) {
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  if (dataExv.rank() == 0)
    return bounds;

  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (std::size_t dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    mlir::Value ext = fir::factory::readExtent(builder, loc, dataExv, dim);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);

    // ub = extent - 1
    mlir::Value ub = builder.create<mlir::arith::SubIOp>(loc, ext, one);
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
             mlir::Value baseAddr) {
  int dimension = 0;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (const auto &subscript : subscripts) {
    if (const auto *triplet{
            std::get_if<Fortran::parser::SubscriptTriplet>(&subscript.u)}) {
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

      const auto &lower{std::get<0>(triplet->t)};
      if (lower) {
        lval = Fortran::semantics::GetIntValue(lower);
        if (lval) {
          if (defaultLb) {
            lbound = builder.createIntegerConstant(loc, idxTy, *lval - 1);
          } else {
            mlir::Value lb = builder.createIntegerConstant(loc, idxTy, *lval);
            lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          }
          asFortran << *lval;
        } else {
          const Fortran::lower::SomeExpr *lexpr =
              Fortran::semantics::GetExpr(*lower);
          mlir::Value lb =
              fir::getBase(converter.genExprValue(loc, *lexpr, stmtCtx));
          lb = builder.createConvert(loc, baseLb.getType(), lb);
          lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          asFortran << lexpr->AsFortran();
        }
      } else {
        lbound = defaultLb ? zero : baseLb;
      }
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
                                 "an OpenMP array section");
            break;
          }
        }
      }
      if (!ubound) {
        extent = fir::factory::readExtent(builder, loc, dataExv, dimension);
        if (defaultLb) {
          // ub = extent - 1
          ubound = builder.create<mlir::arith::SubIOp>(loc, extent, one);
        } else {
          // ub = baseLb + extent - 1
          mlir::Value lbExt =
              builder.create<mlir::arith::AddIOp>(loc, extent, baseLb);
          ubound = builder.create<mlir::arith::SubIOp>(loc, lbExt, one);
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
mlir::Value gatherDataOperandAddrAndBounds(
    Fortran::lower::AbstractConverter &converter, fir::FirOpBuilder &builder,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx, const ObjectType &object,
    mlir::Location operandLocation, std::stringstream &asFortran,
    llvm::SmallVector<mlir::Value> &bounds) {
  mlir::Value baseAddr;

  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto expr{Fortran::semantics::AnalyzeExpr(semanticsContext,
                                                          designator)}) {
              if ((*expr).Rank() > 0 &&
                  Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                      designator)) {
                const auto *arrayElement =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator);
                const auto *dataRef =
                    std::get_if<Fortran::parser::DataRef>(&designator.u);
                fir::ExtendedValue dataExv;
                if (Fortran::parser::Unwrap<
                        Fortran::parser::StructureComponent>(
                        arrayElement->base)) {
                  auto exprBase = Fortran::semantics::AnalyzeExpr(
                      semanticsContext, arrayElement->base);
                  dataExv = converter.genExprAddr(operandLocation, *exprBase,
                                                  stmtCtx);
                  baseAddr = fir::getBase(dataExv);
                  asFortran << (*exprBase).AsFortran();
                } else {
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  baseAddr = getDataOperandBaseAddr(
                      converter, builder, *name.symbol, operandLocation);
                  dataExv = converter.getSymbolExtendedValue(*name.symbol);
                  asFortran << name.ToString();
                }

                if (!arrayElement->subscripts.empty()) {
                  asFortran << '(';
                  bounds = genBoundsOps<BoundsType, BoundsOp>(
                      builder, operandLocation, converter, stmtCtx,
                      arrayElement->subscripts, asFortran, dataExv, baseAddr);
                }
                asFortran << ')';
              } else if (Fortran::parser::Unwrap<
                             Fortran::parser::StructureComponent>(designator)) {
                fir::ExtendedValue compExv =
                    converter.genExprAddr(operandLocation, *expr, stmtCtx);
                baseAddr = fir::getBase(compExv);
                if (fir::unwrapRefType(baseAddr.getType())
                        .isa<fir::SequenceType>())
                  bounds = genBaseBoundsOps<BoundsType, BoundsOp>(
                      builder, operandLocation, converter, compExv, baseAddr);
                asFortran << (*expr).AsFortran();

                if (auto loadOp = mlir::dyn_cast_or_null<fir::LoadOp>(
                        baseAddr.getDefiningOp())) {
                  if (fir::isAllocatableType(loadOp.getType()) ||
                      fir::isPointerType(loadOp.getType()))
                    baseAddr = builder.create<fir::BoxAddrOp>(operandLocation,
                                                              baseAddr);
                }

                // If the component is an allocatable or pointer the result of
                // genExprAddr will be the result of a fir.box_addr operation or
                // a fir.box_addr has been inserted just before.
                // Retrieve the box so we handle it like other descriptor.
                if (auto boxAddrOp = mlir::dyn_cast_or_null<fir::BoxAddrOp>(
                        baseAddr.getDefiningOp())) {
                  baseAddr = boxAddrOp.getVal();
                  bounds = genBoundsOpsFromBox<BoundsType, BoundsOp>(
                      builder, operandLocation, converter, compExv, baseAddr);
                }
              } else {
                // Scalar or full array.
                if (const auto *dataRef{
                        std::get_if<Fortran::parser::DataRef>(&designator.u)}) {
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  fir::ExtendedValue dataExv =
                      converter.getSymbolExtendedValue(*name.symbol);
                  baseAddr = getDataOperandBaseAddr(
                      converter, builder, *name.symbol, operandLocation);
                  if (fir::unwrapRefType(baseAddr.getType())
                          .isa<fir::BaseBoxType>())
                    bounds = genBoundsOpsFromBox<BoundsType, BoundsOp>(
                        builder, operandLocation, converter, dataExv, baseAddr);
                  if (fir::unwrapRefType(baseAddr.getType())
                          .isa<fir::SequenceType>())
                    bounds = genBaseBoundsOps<BoundsType, BoundsOp>(
                        builder, operandLocation, converter, dataExv, baseAddr);
                  asFortran << name.ToString();
                } else { // Unsupported
                  llvm::report_fatal_error(
                      "Unsupported type of OpenACC operand");
                }
              }
            }
          },
          [&](const Fortran::parser::Name &name) {
            baseAddr = getDataOperandBaseAddr(converter, builder, *name.symbol,
                                              operandLocation);
            asFortran << name.ToString();
          }},
      object.u);
  return baseAddr;
}

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DIRECTIVES_COMMON_H
