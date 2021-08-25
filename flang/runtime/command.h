//===-- runtime/command.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_COMMAND_H_
#define FORTRAN_RUNTIME_COMMAND_H_

#include "cpp-type.h"
#include "entry-names.h"

namespace Fortran::runtime {
extern "C" {
// 16.9.51 COMMAND_ARGUMENT_COUNT
//
// Lowering may need to cast the result to match the precision of the default
// integer kind.
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ArgumentCount)();
}
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_COMMAND_H_
