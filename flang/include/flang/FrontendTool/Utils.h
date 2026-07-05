//===--- Utils.h - Misc utilities for the flang front-end --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for various front-end actions
//  which were split from Frontend to minimise Frontend's dependencies.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FRONTEND_FRONTENDTOOL_UTILS_H
#define FORTRAN_FRONTEND_FRONTENDTOOL_UTILS_H

namespace Fortran::frontend {

class CompilerInstance;

/// ExecuteCompilerInvocation - Execute the given actions described by the
/// compiler invocation object in the given compiler instance.
///
/// \return - True on success.
bool executeCompilerInvocation(CompilerInstance *flang);

} // end namespace Fortran::frontend

#endif // FORTRAN_FRONTEND_FRONTENDTOOL_UTILS_H
