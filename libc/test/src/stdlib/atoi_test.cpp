//===-- Unittests for atoi ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AtoiTest.h"

#include "src/stdlib/atoi.h"

#include "test/UnitTest/Test.h"

ATOI_TEST(Atoi, LIBC_NAMESPACE::atoi)
