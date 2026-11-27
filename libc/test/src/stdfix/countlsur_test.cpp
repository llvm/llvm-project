//===-- Unittests for countlsur -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/src/stdfix/CountlsTest.h"

#include "src/stdfix/countlsur.h"

LIST_COUNTLS_TESTS(unsigned fract, LIBC_NAMESPACE::countlsur);
