//===-- include/flang/Runtime/inquiry.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the API for the inquiry intrinsic functions
// that inquire about shape information in arrays: LBOUND and SIZE.

#ifndef FORTRAN_RUNTIME_INQUIRY_H_
#define FORTRAN_RUNTIME_INQUIRY_H_

#include "flang/Runtime/entry-names.h"
#include <cinttypes>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

std::int64_t RTDECL(LboundDim)(const Descriptor &array, int dim,
    const char *sourceFile = nullptr, int line = 0);
void RTDECL(Ubound)(Descriptor &result, const Descriptor &array, int kind,
    const char *sourceFile = nullptr, int line = 0);
std::int64_t RTDECL(Size)(
    const Descriptor &array, const char *sourceFile = nullptr, int line = 0);
std::int64_t RTDECL(SizeDim)(const Descriptor &array, int dim,
    const char *sourceFile = nullptr, int line = 0);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_INQUIRY_H_
