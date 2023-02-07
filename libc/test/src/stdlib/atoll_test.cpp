//===-- Unittests for atoll -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AtoiTest.h"

#include "src/stdlib/atoll.h"

#include "test/UnitTest/Test.h"

ATOI_TEST(Atoll, __llvm_libc::atoll)
