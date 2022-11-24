//===-- ConvertCall.h -- lowering of calls ----------------------*- C++ -*-===//
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
/// Implements the conversion from evaluate::ProcedureRef to FIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERTCALL_H
#define FORTRAN_LOWER_CONVERTCALL_H

#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"

namespace Fortran::lower {

/// Given a call site for which the arguments were already lowered, generate
/// the call and return the result. This function deals with explicit result
/// allocation and lowering if needed. It also deals with passing the host
/// link to internal procedures.
fir::ExtendedValue genCallOpAndResult(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    Fortran::lower::CallerInterface &caller, mlir::FunctionType callSiteType,
    llvm::Optional<mlir::Type> resultType);

/// If \p arg is the address of a function with a denoted host-association tuple
/// argument, then return the host-associations tuple value of the current
/// procedure. Otherwise, return nullptr.
mlir::Value argumentHostAssocs(Fortran::lower::AbstractConverter &converter,
                               mlir::Value arg);

} // namespace Fortran::lower
#endif // FORTRAN_LOWER_CONVERTCALL_H
