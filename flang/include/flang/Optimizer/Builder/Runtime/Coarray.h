//===-- Coarray.h -- generate Coarray intrinsics runtime calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H

#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

// Get the function type for a prif subroutine with a variable number of
// arguments
#define PRIF_FUNCTYPE(...)                                                     \
  mlir::FunctionType::get(builder.getContext(), /*inputs*/ {__VA_ARGS__},      \
                          /*result*/ {})

// Default prefix for subroutines of PRIF compiled with LLVM
#define PRIFNAME_SUB(fmt)                                                      \
  []() {                                                                       \
    std::ostringstream oss;                                                    \
    oss << "prif_" << fmt;                                                     \
    return fir::NameUniquer::doProcedure({"prif"}, {}, oss.str());             \
  }()

/// Generate Call to runtime prif_init
mlir::Value genInitCoarray(fir::FirOpBuilder &builder, mlir::Location loc);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COARRAY_H
