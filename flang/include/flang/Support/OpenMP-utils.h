//===-- include/flang/Support/OpenMP-utils.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_OPENMP_UTILS_H_
#define FORTRAN_SUPPORT_OPENMP_UTILS_H_

#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Semantics/symbol.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace Fortran::common::openmp {
/// Structure holding the information needed to create and bind entry block
/// arguments associated to a single clause.
struct EntryBlockArgsEntry {
  llvm::SmallVector<Fortran::lower::omp::Object> objects;
  llvm::ArrayRef<mlir::Value> vars;

  bool isValid() const {
    // This check allows specifying a smaller number of objects than values
    // because in some case cases a single symbol generates multiple block
    // arguments.
    return objects.size() <= vars.size();
  }

  llvm::SmallVector<const Fortran::semantics::Symbol *> getSyms() const {
    llvm::SmallVector<const Fortran::semantics::Symbol *> syms;
    syms.reserve(objects.size());
    llvm::transform(objects, std::back_inserter(syms),
        [](const Fortran::lower::omp::Object &object) { return object.sym(); });
    return syms;
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

  llvm::SmallVector<const semantics::Symbol *> getSyms() const {
    llvm::SmallVector<const semantics::Symbol *> syms;
    auto appendSyms = [&syms](const EntryBlockArgsEntry &entry) {
      syms.reserve(syms.size() + entry.objects.size());
      llvm::transform(entry.objects, std::back_inserter(syms),
          [](const Fortran::lower::omp::Object &object) {
            return object.sym();
          });
    };
    appendSyms(hasDeviceAddr);
    appendSyms(inReduction);
    appendSyms(map);
    appendSyms(priv);
    appendSyms(reduction);
    appendSyms(taskReduction);
    appendSyms(useDeviceAddr);
    appendSyms(useDevicePtr);
    return syms;
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
