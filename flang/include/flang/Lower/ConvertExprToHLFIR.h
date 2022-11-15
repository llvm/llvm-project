//===-- Lower/ConvertExprToHLFIR.h -- lowering of expressions ----*- C++-*-===//
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
/// Implements the conversion from Fortran::evaluate::Expr trees to HLFIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERTEXPRTOHLFIR_H
#define FORTRAN_LOWER_CONVERTEXPRTOHLFIR_H

#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"

namespace mlir {
class Location;
} // namespace mlir

namespace Fortran::lower {

class AbstractConverter;
class StatementContext;
class SymMap;

hlfir::EntityWithAttributes
convertExprToHLFIR(mlir::Location loc, Fortran::lower::AbstractConverter &,
                   const Fortran::lower::SomeExpr &, Fortran::lower::SymMap &,
                   Fortran::lower::StatementContext &);
} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTEXPRTOHLFIR_H
