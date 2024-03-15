//===-- RandUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_RANDUTILS_H
#define LLVM_LIBC_TEST_SRC_MATH_RANDUTILS_H

namespace LIBC_NAMESPACE {
namespace testutils {

// Wrapper for std::rand.
int rand();

} // namespace testutils
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_TEST_SRC_MATH_RANDUTILS_H
