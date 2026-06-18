//===-- Platform specific defines for the unittest library ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_PLATFORMDEFS_H
#define LLVM_LIBC_TEST_UNITTEST_PLATFORMDEFS_H

#define LIBC_TEST_UNIT 1
#define LIBC_TEST_HERMETIC 2

#define CONCAT_HELPER(a, b) a ## b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

#define CHECK_TEST_TYPE(type) CONCAT(LIBC_TEST_, type)

#if !defined(_WIN32) && (CHECK_TEST_TYPE(LIBC_TEST) != LIBC_TEST_HERMETIC)
#define ENABLE_SUBPROCESS_TESTS
#endif

#endif // LLVM_LIBC_TEST_UNITTEST_PLATFORMDEFS_H
