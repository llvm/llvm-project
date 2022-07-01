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

/// Generate a call to the GetCommandArgument runtime function which implements
/// the GET_COMMAND_ARGUMENT intrinsic.
/// \p value, \p length and \p errmsg must be fir.box that can be absent (but
/// not null mlir values). The status value is returned.
mlir::Value genGetCommandArgument(fir::FirOpBuilder &, mlir::Location,
                                  mlir::Value number, mlir::Value value,
                                  mlir::Value length, mlir::Value errmsg);

/// Generate a call to EnvVariableValue runtime function which implements
/// the part of GET_ENVIRONMENT_ARGUMENT related to VALUE, ERRMSG, and STATUS.
/// \p value and \p errmsg must be fir.box that can be absent (but not null
/// mlir values). The status value is returned. \p name must be a fir.box.
/// and \p trimName a boolean value.
mlir::Value genEnvVariableValue(fir::FirOpBuilder &, mlir::Location,
                                mlir::Value name, mlir::Value value,
                                mlir::Value trimName, mlir::Value errmsg);

/// Generate a call to EnvVariableLength runtime function which implements
/// the part of GET_ENVIRONMENT_ARGUMENT related to LENGTH.
/// It returns the length of the \p number command arguments.
/// \p name must be a fir.box and \p trimName a boolean value.
mlir::Value genEnvVariableLength(fir::FirOpBuilder &, mlir::Location,
                                 mlir::Value name, mlir::Value trimName);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
