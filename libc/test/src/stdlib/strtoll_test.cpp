//===-- Unittests for strtoll ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strtoll.h"

#include "test/UnitTest/Test.h"

#include "StrtolTest.h"

STRTOL_TEST(Strtoll, __llvm_libc::strtoll)
