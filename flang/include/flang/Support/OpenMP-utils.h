//===-- include/flang/Support/OpenMP-utils.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_OPENMP_UTILS_H_
#define FORTRAN_SUPPORT_OPENMP_UTILS_H_

#include "flang/Semantics/symbol.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"

namespace Fortran::common::openmp {
/// Structure holding the information needed to create and bind entry block
/// arguments associated to a single clause.
struct EntryBlockArgsEntry {
  llvm::ArrayRef<const Fortran::semantics::Symbol *> syms;
  llvm::ArrayRef<mlir::Value> vars;

  bool isValid() const {
    // This check allows specifying a smaller number of symbols than values
    // because in some case cases a single symbol generates multiple block
    // arguments.
    return syms.size() <= vars.size();
  }
};

/// Structure holding the information needed to create and bind entry block
/// arguments associated to all clauses that can define them.
struct EntryBlockArgs {
  EntryBlockArgsEntry hasDeviceAddr;
  llvm::ArrayRef<mlir::Value> hostEvalVars;
  EntryBlockArgsEntry inReduction;
  EntryBlockArgsEntry map;
  EntryBlockArgsEntry priv;
  EntryBlockArgsEntry reduction;
  EntryBlockArgsEntry taskReduction;
  EntryBlockArgsEntry useDeviceAddr;
  EntryBlockArgsEntry useDevicePtr;

  bool isValid() const {
    return hasDeviceAddr.isValid() && inReduction.isValid() && map.isValid() &&
        priv.isValid() && reduction.isValid() && taskReduction.isValid() &&
        useDeviceAddr.isValid() && useDevicePtr.isValid();
  }

  auto getSyms() const {
    return llvm::concat<const semantics::Symbol *const>(hasDeviceAddr.syms,
        inReduction.syms, map.syms, priv.syms, reduction.syms,
        taskReduction.syms, useDeviceAddr.syms, useDevicePtr.syms);
  }

  auto getVars() const {
    return llvm::concat<const mlir::Value>(hasDeviceAddr.vars, hostEvalVars,
        inReduction.vars, map.vars, priv.vars, reduction.vars,
        taskReduction.vars, useDeviceAddr.vars, useDevicePtr.vars);
  }
};

/// Create an entry block for the given region, including the clause-defined
/// arguments specified.
///
/// \param [in] builder - MLIR operation builder.
/// \param [in]    args - entry block arguments information for the given
///                       operation.
/// \param [in]  region - Empty region in which to create the entry block.
mlir::Block *genEntryBlock(
    mlir::OpBuilder &builder, const EntryBlockArgs &args, mlir::Region &region);
} // namespace Fortran::common::openmp

#endif // FORTRAN_SUPPORT_OPENMP_UTILS_H_
