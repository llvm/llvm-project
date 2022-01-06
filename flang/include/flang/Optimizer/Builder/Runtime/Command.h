//===-- Command.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate call to COMMAND_ARGUMENT_COUNT intrinsic runtime routine.
mlir::Value genCommandArgumentCount(fir::FirOpBuilder &, mlir::Location);

/// Generate call to GET_COMMAND_ARGUMENT intrinsic runtime routine.
/// Note that GET_COMMAND_ARGUMENT intrinsic is split between 2 functions in
/// implementation; ArgumentValue and ArgumentLength. So we handle each
/// seperately.
void genGetCommandArgument(fir::FirOpBuilder &, mlir::Location,
                           mlir::Value number, mlir::Value value,
                           mlir::Value length, mlir::Value status,
                           mlir::Value errmsg);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
