//===-- Implementation header for canonicalizef ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_CANONICALIZEF_H
#define LLVM_LIBC_SRC_MATH_CANONICALIZEF_H

namespace LIBC_NAMESPACE {

int canonicalizef(float *cx, const float *x);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_CANONICALIZEF_H
