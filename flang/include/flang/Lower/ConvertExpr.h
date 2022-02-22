//===-- Lower/ConvertExpr.h -- lowering of expressions ----------*- C++ -*-===//
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
/// Implements the conversion from Fortran::evaluate::Expr trees to FIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERTEXPR_H
#define FORTRAN_LOWER_CONVERTEXPR_H

#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace mlir {
class Location;
}

namespace Fortran::evaluate {
template <typename>
class Expr;
struct SomeType;
} // namespace Fortran::evaluate

namespace Fortran::lower {

class AbstractConverter;
class SymMap;
using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;

/// Create an extended expression value.
fir::ExtendedValue createSomeExtendedExpression(mlir::Location loc,
                                                AbstractConverter &converter,
                                                const SomeExpr &expr,
                                                SymMap &symMap);

/// Create an extended expression address.
fir::ExtendedValue createSomeExtendedAddress(mlir::Location loc,
                                             AbstractConverter &converter,
                                             const SomeExpr &expr,
                                             SymMap &symMap);

// Attribute for an alloca that is a trivial adaptor for converting a value to
// pass-by-ref semantics for a VALUE parameter. The optimizer may be able to
// eliminate these.
inline mlir::NamedAttribute getAdaptToByRefAttr(fir::FirOpBuilder &builder) {
  return {mlir::StringAttr::get(builder.getContext(),
                                fir::getAdaptToByRefAttrName()),
          builder.getUnitAttr()};
}

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTEXPR_H
