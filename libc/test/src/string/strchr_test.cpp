//===-- Unittests for strchr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StrchrTest.h"

#include "src/string/strchr.h"
#include "test/UnitTest/Test.h"

STRCHR_TEST(Strchr, LIBC_NAMESPACE::strchr)
