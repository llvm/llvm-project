//===- ConvertProcedureDesignator.h -- Procedure Designators ----*- C++ -*-===//
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
/// Lowering of evaluate::ProcedureDesignator to FIR and HLFIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERT_PROCEDURE_DESIGNATOR_H
#define FORTRAN_LOWER_CONVERT_PROCEDURE_DESIGNATOR_H

namespace mlir {
class Location;
}
namespace fir {
class ExtendedValue;
}
namespace hlfir {
class EntityWithAttributes;
}
namespace Fortran::evaluate {
struct ProcedureDesignator;
}

namespace Fortran::lower {
class AbstractConverter;
class StatementContext;
class SymMap;

/// Lower a procedure designator to a fir::ExtendedValue that can be a
/// fir::CharBoxValue for character procedure designator (the CharBoxValue
/// length carries the result length if it is known).
fir::ExtendedValue convertProcedureDesignator(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx);

/// Lower a procedure designator to a !fir.boxproc<()->() or
/// tuple<!fir.boxproc<()->(), len>.
hlfir::EntityWithAttributes convertProcedureDesignatorToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx);

} // namespace Fortran::lower
#endif // FORTRAN_LOWER_CONVERT_PROCEDURE_DESIGNATOR_H
