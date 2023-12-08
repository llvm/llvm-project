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
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include <optional>

namespace Fortran::lower {

/// Given a call site for which the arguments were already lowered, generate
/// the call and return the result. This function deals with explicit result
/// allocation and lowering if needed. It also deals with passing the host
/// link to internal procedures.
/// \p isElemental must be set to true if elemental call is being produced.
/// It is only used for HLFIR.
fir::ExtendedValue genCallOpAndResult(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    Fortran::lower::CallerInterface &caller, mlir::FunctionType callSiteType,
    std::optional<mlir::Type> resultType, bool isElemental = false);

/// If \p arg is the address of a function with a denoted host-association tuple
/// argument, then return the host-associations tuple value of the current
/// procedure. Otherwise, return nullptr.
mlir::Value argumentHostAssocs(Fortran::lower::AbstractConverter &converter,
                               mlir::Value arg);

/// Is \p procRef an intrinsic module procedure that should be lowered as
/// intrinsic procedures (with Optimizer/Builder/IntrinsicCall.h)?
bool isIntrinsicModuleProcRef(const Fortran::evaluate::ProcedureRef &procRef);

/// Lower a ProcedureRef to HLFIR. If this is a function call, return the
/// lowered result value. Return nothing otherwise.
std::optional<hlfir::EntityWithAttributes> convertCallToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const evaluate::ProcedureRef &procRef, std::optional<mlir::Type> resultType,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx);

void convertUserDefinedAssignmentToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const evaluate::ProcedureRef &procRef, hlfir::Entity lhs, hlfir::Entity rhs,
    Fortran::lower::SymMap &symMap);
} // namespace Fortran::lower
#endif // FORTRAN_LOWER_CONVERTCALL_H
