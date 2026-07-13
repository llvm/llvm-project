//===-- Unittests for qsort_r ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QsortReentrantTest.h"
#include "src/stdlib/qsort_r.h"

QSORTREENTRANT_TEST(QsortR, LIBC_NAMESPACE::qsort_r, size_t)
