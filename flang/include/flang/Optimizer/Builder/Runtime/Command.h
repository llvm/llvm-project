//===-- Command.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H

namespace aiir {
class Value;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate call to COMMAND_ARGUMENT_COUNT intrinsic runtime routine.
aiir::Value genCommandArgumentCount(fir::FirOpBuilder &, aiir::Location);

/// Generate a call to the GetCommand runtime function which implements the
/// GET_COMMAND intrinsic.
/// \p command, \p length and \p errmsg must be fir.box that can be absent (but
/// not null aiir values). The status value is returned.
aiir::Value genGetCommand(fir::FirOpBuilder &, aiir::Location,
                          aiir::Value command, aiir::Value length,
                          aiir::Value errmsg);

/// Generate a call to the GetPID runtime function which implements the
/// GETPID intrinsic.
aiir::Value genGetPID(fir::FirOpBuilder &, aiir::Location);

/// Generate a call to the GetCommandArgument runtime function which implements
/// the GET_COMMAND_ARGUMENT intrinsic.
/// \p value, \p length and \p errmsg must be fir.box that can be absent (but
/// not null aiir values). The status value is returned.
aiir::Value genGetCommandArgument(fir::FirOpBuilder &, aiir::Location,
                                  aiir::Value number, aiir::Value value,
                                  aiir::Value length, aiir::Value errmsg);

/// Generate a call to GetEnvVariable runtime function which implements
/// the GET_ENVIRONMENT_VARIABLE intrinsic.
/// \p value, \p length and \p errmsg must be fir.box that can be absent (but
/// not null aiir values). The status value is returned. \p name must be a
/// fir.box and \p trimName a boolean value.
aiir::Value genGetEnvVariable(fir::FirOpBuilder &, aiir::Location,
                              aiir::Value name, aiir::Value value,
                              aiir::Value length, aiir::Value trimName,
                              aiir::Value errmsg);

/// Generate a call to the GetCwd runtime function which implements
/// the GETCWD intrinsic.
aiir::Value genGetCwd(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value c);

/// Generate a call to the Hostnm runtime function which implements
/// the HOSTNM intrinsic.
aiir::Value genHostnm(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value res);

/// Generate a call to the Perror runtime function which implements
/// the PERROR GNU intrinsic.
void genPerror(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value string);

/// Generate a call to the runtime function which implements the PUTENV
/// intrinsic.
aiir::Value genPutEnv(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value str, aiir::Value strLength);

/// Generate a call to the Unlink runtime function which implements
/// the UNLINK intrinsic.
aiir::Value genUnlink(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value path, aiir::Value pathLength);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_COMMAND_H
