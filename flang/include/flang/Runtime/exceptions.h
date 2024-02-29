//===-- include/flang/Runtime/exceptions.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Map Fortran ieee_arithmetic module exceptions to fenv.h exceptions.

#ifndef FORTRAN_RUNTIME_EXCEPTIONS_H_
#define FORTRAN_RUNTIME_EXCEPTIONS_H_

#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/magic-numbers.h"
#include <cinttypes>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

// Map a (single) IEEE_FLAG_TYPE exception value to a libm fenv.h value.
// This could be extended to handle sets of exceptions, but there is no
// current use case for that. This mapping is done at runtime to support
// cross compilation.
std::int32_t RTNAME(MapException)(std::int32_t except);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_EXCEPTIONS_H_
