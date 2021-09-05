//===-- include/flang/Runtime/command.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_COMMAND_H_
#define FORTRAN_RUNTIME_COMMAND_H_

#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
class Descriptor;

extern "C" {
// 16.9.51 COMMAND_ARGUMENT_COUNT
//
// Lowering may need to cast the result to match the precision of the default
// integer kind.
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ArgumentCount)();

// 16.9.83 GET_COMMAND_ARGUMENT
// We're breaking up the interface into several different functions, since most
// of the parameters are optional.

// Try to get the value of the n'th argument.
// Returns a STATUS as described in the standard.
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ArgumentValue)(
    CppTypeFor<TypeCategory::Integer, 4> n, const Descriptor *value,
    const Descriptor *errmsg);

// Try to get the significant length of the n'th argument.
// Returns 0 if it doesn't manage.
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ArgumentLength)(
    CppTypeFor<TypeCategory::Integer, 4> n);
}
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_COMMAND_H_
