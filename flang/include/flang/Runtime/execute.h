//===-- include/flang/Runtime/command.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_EXECUTE_H_
#define FORTRAN_RUNTIME_EXECUTE_H_

#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
class Descriptor;

extern "C" {

// 16.9.83 EXECUTE_COMMAND_LINE
// Execute a command line.
// Returns a EXITSTAT, CMDSTAT, and CMDMSG as described in the standard.
void RTNAME(ExecuteCommandLine)(const Descriptor &command, bool wait = true,
    const Descriptor *exitstat = nullptr, const Descriptor *cmdstat = nullptr,
    const Descriptor *cmdmsg = nullptr, const char *sourceFile = nullptr,
    int line = 0);
}
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_EXECUTE_H_
