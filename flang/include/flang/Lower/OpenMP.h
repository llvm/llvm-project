//===-- Lower/OpenMP.h -- lower Open MP directives --------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_OPENMP_H
#define FORTRAN_LOWER_OPENMP_H

namespace mlir {
class Operation;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran {
namespace parser {
struct OpenMPConstruct;
} // namespace parser

namespace lower {

// Generate the OpenMP terminator for Operation at Location.
void genOpenMPTerminator(fir::FirOpBuilder &, mlir::Operation *,
                         mlir::Location);

bool isOpenMPTargetConstruct(const parser::OpenMPConstruct &);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMP_H
