//===-- Allocatable.h -- Allocatable statements lowering ------------------===//
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

#ifndef FORTRAN_LOWER_ALLOCATABLE_H
#define FORTRAN_LOWER_ALLOCATABLE_H

#include "flang/Optimizer/Builder/MutableBox.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Value;
class ValueRange;
class Location;
} // namespace mlir

namespace fir {
class MutableBoxValue;
}

namespace Fortran::parser {
struct AllocateStmt;
struct DeallocateStmt;
} // namespace Fortran::parser

namespace Fortran::evaluate {
template <typename T>
class Expr;
struct SomeType;
} // namespace Fortran::evaluate

namespace Fortran::lower {
class AbstractConverter;
class StatementContext;

namespace pft {
struct Variable;
}

/// Lower an allocate statement to fir.
void genAllocateStmt(Fortran::lower::AbstractConverter &,
                     const Fortran::parser::AllocateStmt &, mlir::Location);

/// Lower a deallocate statement to fir.
void genDeallocateStmt(Fortran::lower::AbstractConverter &,
                       const Fortran::parser::DeallocateStmt &, mlir::Location);

/// Create a MutableBoxValue for an allocatable or pointer entity.
/// If the variables is a local variable that is not a dummy, it will be
/// initialized to unallocated/diassociated status.
fir::MutableBoxValue createMutableBox(Fortran::lower::AbstractConverter &,
                                      mlir::Location,
                                      const Fortran::lower::pft::Variable &var,
                                      mlir::Value boxAddr,
                                      mlir::ValueRange nonDeferredParams);

/// Update a MutableBoxValue to describe the entity designated by the expression
/// \p source. This version takes care of \p source lowering.
/// If \lbounds is not empty, it is used to defined the MutableBoxValue
/// lower bounds, otherwise, the lower bounds from \p source are used.
void associateMutableBox(
    Fortran::lower::AbstractConverter &, mlir::Location,
    const fir::MutableBoxValue &,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &source,
    mlir::ValueRange lbounds, Fortran::lower::StatementContext &);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_ALLOCATABLE_H
