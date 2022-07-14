//===--- Round floating point to nearest integer on aarch64 -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_NEAREST_INTEGER_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_NEAREST_INTEGER_H

#include "src/__support/architectures.h"

#if !defined(LLVM_LIBC_ARCH_AARCH64)
#error "Invalid include"
#endif

namespace __llvm_libc {
namespace fputil {

static inline double nearest_integer(double x) {
  double result;
  __asm__ __volatile__("frintn %d0, %d1\n\t" : "=w"(result) : "w"(x));
  return result;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_AARCH64_NEAREST_INTEGER_H
