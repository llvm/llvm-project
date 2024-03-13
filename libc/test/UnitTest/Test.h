//===-- Header selector for libc unittests ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_TEST_H
#define LLVM_LIBC_TEST_UNITTEST_TEST_H

// This macro takes a file name and returns a value implicitly castable to
// a const char*. That const char* is the path to a file with the provided name
// in a directory where the test is allowed to write. By default it writes
// directly to the filename provided, but implementations are allowed to
// redefine it as necessary.
#define libc_make_test_file_path(file_name) (file_name)

#if defined(LIBC_COPT_TEST_USE_FUCHSIA)
#include "FuchsiaTest.h"
#elif defined(LIBC_COPT_TEST_USE_PIGWEED)
#include "PigweedTest.h"
#else
#include "LibcTest.h"
#endif

#endif // LLVM_LIBC_TEST_UNITTEST_TEST_H
