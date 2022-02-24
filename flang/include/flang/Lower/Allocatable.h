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

namespace Fortran::lower {
class AbstractConverter;

namespace pft {
struct Variable;
}

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
