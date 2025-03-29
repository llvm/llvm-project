//===-- Command.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXECUTE_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXECUTE_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate a call to the ExecuteCommandLine runtime function which implements
/// the GET_EXECUTE_ARGUMENT intrinsic.
/// \p wait must be bool that can be absent.
/// \p exitstat, \p cmdstat and \p cmdmsg must be fir.box that can be
/// absent (but not null mlir values). The status exitstat and cmdstat are
/// returned, along with the message cmdmsg.
void genExecuteCommandLine(fir::FirOpBuilder &, mlir::Location,
                           mlir::Value command, mlir::Value wait,
                           mlir::Value exitstat, mlir::Value cmdstat,
                           mlir::Value cmdmsg);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXECUTE_H
