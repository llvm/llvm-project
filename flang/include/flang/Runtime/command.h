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

namespace Fortran::runtime {
class Descriptor;

extern "C" {
// 16.9.51 COMMAND_ARGUMENT_COUNT
//
// Lowering may need to cast the result to match the precision of the default
// integer kind.
std::int32_t RTNAME(ArgumentCount)();

// 16.9.83 GET_COMMAND_ARGUMENT
// We're breaking up the interface into several different functions, since most
// of the parameters are optional.

// Try to get the value of the n'th argument.
// Returns a STATUS as described in the standard.
std::int32_t RTNAME(ArgumentValue)(
    std::int32_t n, const Descriptor *value, const Descriptor *errmsg);

// Try to get the significant length of the n'th argument.
// Returns 0 if it doesn't manage.
std::int64_t RTNAME(ArgumentLength)(std::int32_t n);

// 16.9.84 GET_ENVIRONMENT_VARIABLE
// We're breaking up the interface into several different functions, since most
// of the parameters are optional.

// Try to get the value of the environment variable specified by NAME.
// Returns a STATUS as described in the standard.
std::int32_t RTNAME(EnvVariableValue)(const Descriptor &name,
    const Descriptor *value = nullptr, bool trim_name = true,
    const Descriptor *errmsg = nullptr);

// Try to get the significant length of the environment variable specified by
// NAME. Returns 0 if it doesn't manage.
std::int64_t RTNAME(EnvVariableLength)(
    const Descriptor &name, bool trim_name = true);
}
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_COMMAND_H_
