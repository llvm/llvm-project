//===-- Header for setting up the Pigweed tests -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_PIGWEEDTEST_H
#define LLVM_LIBC_UTILS_UNITTEST_PIGWEEDTEST_H

#include <gtest/gtest.h>

namespace LIBC_NAMESPACE::testing {
using Test = ::testing::Test;
}

#endif // LLVM_LIBC_UTILS_UNITTEST_PIGWEEDTEST_H
