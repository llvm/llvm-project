//===-- Utilities for triple-double data type. ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_TRIPLEDOUBLE_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_TRIPLEDOUBLE_H

namespace __llvm_libc::fputil {

struct TripleDouble {
  double lo = 0.0;
  double mid = 0.0;
  double hi = 0.0;
};

} // namespace __llvm_libc::fputil

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_TRIPLEDOUBLE_H
