//===-- Implementation header for cosf --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_COSF_H
#define LLVM_LIBC_SRC_MATH_COSF_H

namespace __llvm_libc {
#if defined(__CLANG_GPU_APPROX_TRANSCENDENTALS__)
namespace fast {
    float cosf(float x);
}
using fast::cosf;
#else
float cosf(float x);
#endif

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_COSF_H
