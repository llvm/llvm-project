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
class Value;
class Type;
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
namespace Fortran::semantics {
class Symbol;
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

/// Generate initialization for procedure pointer to procedure target.
mlir::Value
convertProcedureDesignatorInitialTarget(Fortran::lower::AbstractConverter &,
                                        mlir::Location,
                                        const Fortran::semantics::Symbol &sym);

/// Given the value of a "PASS" actual argument \p passedArg and the
/// evaluate::ProcedureDesignator for the call, address and dereference
/// the argument's procedure pointer component that must be called.
mlir::Value derefPassProcPointerComponent(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc, mlir::Value passedArg,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx);
} // namespace Fortran::lower
#endif // FORTRAN_LOWER_CONVERT_PROCEDURE_DESIGNATOR_H
