//===-- NVPTX specific definitions for math support -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GPU_NVPTX_H
#define LLVM_LIBC_SRC_MATH_GPU_NVPTX_H

#include "declarations.h"

#include "src/__support/macros/attributes.h"

namespace __llvm_libc {
namespace internal {

LIBC_INLINE double sin(double x) { return __nv_sin(x); }

} // namespace internal
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GPU_NVPTX_H
