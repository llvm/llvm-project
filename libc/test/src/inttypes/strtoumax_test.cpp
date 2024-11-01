//===-- Unittests for strtoumax -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/inttypes/strtoumax.h"

#include "test/UnitTest/Test.h"

#include "test/src/stdlib/StrtolTest.h"

STRTOL_TEST(Strtoumax, __llvm_libc::strtoumax)
