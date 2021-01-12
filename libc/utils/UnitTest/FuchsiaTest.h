//===-- Header for setting up the Fuchsia tests -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_FUCHSIATEST_H
#define LLVM_LIBC_UTILS_UNITTEST_FUCHSIATEST_H

#include <zxtest/zxtest.h>
// isascii is being undef'd because Fuchsia's headers define a macro for
// isascii. that macro causes errors when isascii_test.cpp references
// __llvm_libc::isascii since the macro is applied first.
#ifdef isascii
#undef isascii
#endif

#endif // LLVM_LIBC_UTILS_UNITTEST_FUCHSIATEST_H
