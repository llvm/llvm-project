//===-- Header selector for libc unittests ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_TEST_H
#define LLVM_LIBC_UTILS_UNITTEST_TEST_H

#ifdef LIBC_COPT_TEST_USE_FUCHSIA
#include "FuchsiaTest.h"
#else
#include "LibcTest.h"
#endif

#endif // LLVM_LIBC_UTILS_UNITTEST_TEST_H
