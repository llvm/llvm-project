//===-- include/flang/Runtime/command.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_COMMAND_H_
#define FORTRAN_RUNTIME_COMMAND_H_

#include "flang/Runtime/entry-names.h"
#include <cstdint>

#ifdef _WIN32
// On Windows* OS GetCurrentProcessId returns DWORD aka uint32_t
typedef std::uint32_t pid_t;
#else
#include "sys/types.h" //pid_t
#endif

namespace Fortran::runtime {
class Descriptor;

extern "C" {
// 16.9.51 COMMAND_ARGUMENT_COUNT
//
// Lowering may need to cast the result to match the precision of the default
// integer kind.
std::int32_t RTNAME(ArgumentCount)();

// Calls getpid()
pid_t RTNAME(GetPID)();

// 16.9.82 GET_COMMAND
// Try to get the value of the whole command. All of the parameters are
// optional.
// Return a STATUS as described in the standard.
std::int32_t RTNAME(GetCommand)(const Descriptor *command = nullptr,
    const Descriptor *length = nullptr, const Descriptor *errmsg = nullptr,
    const char *sourceFile = nullptr, int line = 0);

// 16.9.83 GET_COMMAND_ARGUMENT
// Try to get the value of the n'th argument.
// Returns a STATUS as described in the standard.
std::int32_t RTNAME(GetCommandArgument)(std::int32_t n,
    const Descriptor *argument = nullptr, const Descriptor *length = nullptr,
    const Descriptor *errmsg = nullptr, const char *sourceFile = nullptr,
    int line = 0);

// 16.9.84 GET_ENVIRONMENT_VARIABLE
// Try to get the value of the environment variable specified by NAME.
// Returns a STATUS as described in the standard.
std::int32_t RTNAME(GetEnvVariable)(const Descriptor &name,
    const Descriptor *value = nullptr, const Descriptor *length = nullptr,
    bool trim_name = true, const Descriptor *errmsg = nullptr,
    const char *sourceFile = nullptr, int line = 0);

// Calls getcwd()
std::int32_t RTNAME(GetCwd)(
    const Descriptor &cwd, const char *sourceFile, int line);
}
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_COMMAND_H_
