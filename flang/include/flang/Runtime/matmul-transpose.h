//===-- include/flang/Runtime/matmul-transpose.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// API for optimised MATMUL(TRANSPOSE(a), b)

#ifndef FORTRAN_RUNTIME_MATMUL_TRANSPOSE_H_
#define FORTRAN_RUNTIME_MATMUL_TRANSPOSE_H_
#include "flang/Runtime/entry-names.h"
namespace Fortran::runtime {
class Descriptor;
extern "C" {

// The most general MATMUL(TRANSPOSE()).  All type and shape information is
// taken from the arguments' descriptors, and the result is dynamically
// allocated.
void RTNAME(MatmulTranspose)(Descriptor &, const Descriptor &,
    const Descriptor &, const char *sourceFile = nullptr, int line = 0);

// A non-allocating variant; the result's descriptor must be established
// and have a valid base address.
void RTNAME(MatmulTransposeDirect)(const Descriptor &, const Descriptor &,
    const Descriptor &, const char *sourceFile = nullptr, int line = 0);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_MATMUL_TRANSPOSE_H_
