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

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace Fortran::parser {
struct AllocateStmt;
struct DeallocateStmt;
} // namespace Fortran::parser

namespace Fortran::lower {
class AbstractConverter;
namespace pft {
class Variable;
}

/// Generate fir to initialize the box (descriptor) of an allocatable variable.
/// Initialization of such box has to be done at the beginning of the variable
/// lifetime.
/// The memory address of the box to be initialized must be provided as an
/// input.
void genAllocatableInit(Fortran::lower::AbstractConverter &,
                        const Fortran::lower::pft::Variable &,
                        mlir::Value boxAddress);

/// Lower an allocate statement to fir.
void genAllocateStmt(Fortran::lower::AbstractConverter &,
                     const Fortran::parser::AllocateStmt &, mlir::Location);

/// Lower a deallocate statement to fir.
void genDeallocateStmt(Fortran::lower::AbstractConverter &,
                       const Fortran::parser::DeallocateStmt &, mlir::Location);
} // namespace Fortran::lower
