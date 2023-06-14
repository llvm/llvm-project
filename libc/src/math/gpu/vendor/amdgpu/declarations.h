//===-- AMDGPU specific declarations for math support ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GPU_AMDGPU_DECLARATIONS_H
#define LLVM_LIBC_SRC_MATH_GPU_AMDGPU_DECLARATIONS_H

namespace __llvm_libc {

extern "C" {
double __ocml_sin_f64(double);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GPU_AMDGPU_DECLARATIONS_H
