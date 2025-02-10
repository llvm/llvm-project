//===-- Unittests for countlsuhr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CountlsTest.h"

#include "src/stdfix/countlsuhr.h"

LIST_COUNTLS_TESTS(unsigned short fract, LIBC_NAMESPACE::countlsuhr);
