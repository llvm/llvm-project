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
} // namespace fir

namespace Fortran::parser {
struct AllocateStmt;
struct DeallocateStmt;
} // namespace Fortran::parser

namespace Fortran::lower {
class AbstractConverter;

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
/// initialized to unallocated/disassociated status.
fir::MutableBoxValue createMutableBox(Fortran::lower::AbstractConverter &,
                                      mlir::Location,
                                      const Fortran::lower::pft::Variable &var,
                                      mlir::Value boxAddr,
                                      mlir::ValueRange nonDeferredParams);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_ALLOCATABLE_H
