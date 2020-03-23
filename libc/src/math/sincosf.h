//===-- Implementation header for sincosf -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_SINCOSF_H
#define LLVM_LIBC_SRC_MATH_SINCOSF_H

namespace __llvm_libc {

void sincosf(float x, float *sinx, float *cosx);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_SINCOSF_H
