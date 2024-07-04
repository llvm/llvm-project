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
#include "flang/Common/float128.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/entry-names.h"
namespace Fortran::runtime {
class Descriptor;
extern "C" {

// The most general MATMUL(TRANSPOSE()).  All type and shape information is
// taken from the arguments' descriptors, and the result is dynamically
// allocated.
void RTDECL(MatmulTranspose)(Descriptor &, const Descriptor &,
    const Descriptor &, const char *sourceFile = nullptr, int line = 0);

// A non-allocating variant; the result's descriptor must be established
// and have a valid base address.
void RTDECL(MatmulTransposeDirect)(const Descriptor &, const Descriptor &,
    const Descriptor &, const char *sourceFile = nullptr, int line = 0);

// MATMUL(TRANSPOSE()) versions specialized by the categories of the operand
// types. The KIND and shape information is taken from the argument's
// descriptors.
#define MATMUL_INSTANCE(XCAT, XKIND, YCAT, YKIND) \
  void RTDECL(MatmulTranspose##XCAT##XKIND##YCAT##YKIND)(Descriptor & result, \
      const Descriptor &x, const Descriptor &y, const char *sourceFile, \
      int line);
#define MATMUL_DIRECT_INSTANCE(XCAT, XKIND, YCAT, YKIND) \
  void RTDECL(MatmulTransposeDirect##XCAT##XKIND##YCAT##YKIND)( \
      Descriptor & result, const Descriptor &x, const Descriptor &y, \
      const char *sourceFile, int line);

#define MATMUL_FORCE_ALL_TYPES 0

#include "matmul-instances.inc"

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_MATMUL_TRANSPOSE_H_
