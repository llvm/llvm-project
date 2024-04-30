//===-- Main.h - generate main runtime API calls ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_MAIN_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_MAIN_H

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
class GlobalOp;
} // namespace fir

namespace fir::runtime {

void genMain(fir::FirOpBuilder &builder, mlir::Location loc,
             fir::GlobalOp &env);

}

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_MAIN_H
