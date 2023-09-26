//===-- Unittests for rindex ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StrchrTest.h"

#include "src/string/rindex.h"
#include "test/UnitTest/Test.h"

STRRCHR_TEST(Rindex, LIBC_NAMESPACE::rindex)
