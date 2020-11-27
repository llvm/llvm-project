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
class ValueRange;
class Type;
class Location;
} // namespace mlir

namespace Fortran::parser {
struct AllocateStmt;
struct DeallocateStmt;
} // namespace Fortran::parser

namespace Fortran::lower {
class AbstractConverter;
class FirOpBuilder;

/// Create a fir.box of type \p boxType that can be used to initialize an
/// allocatable variable. Initialization of such variable has to be done at the
/// beginning of the variable lifetime by storing the created box in the memory
/// for the variable box.
/// \p nonDeferredParams must provide the non deferred length parameters so that
/// they can already be placed in the unallocated box (inquiries about these
/// parameters are legal even in unallocated state).
mlir::Value createUnallocatedBox(Fortran::lower::FirOpBuilder &builder,
                                 mlir::Location loc, mlir::Type boxType,
                                 mlir::ValueRange nonDeferredParams);

/// Lower an allocate statement to fir.
void genAllocateStmt(Fortran::lower::AbstractConverter &,
                     const Fortran::parser::AllocateStmt &, mlir::Location);

/// Lower a deallocate statement to fir.
void genDeallocateStmt(Fortran::lower::AbstractConverter &,
                       const Fortran::parser::DeallocateStmt &, mlir::Location);
} // namespace Fortran::lower
