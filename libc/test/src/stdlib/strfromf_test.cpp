//===-- Unittests for strfromf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StrfromTest.h"
#include "src/stdlib/strfromf.h"
#include "test/UnitTest/Test.h"

STRFROM_TEST(float, StrFromf, LIBC_NAMESPACE::strfromf)
