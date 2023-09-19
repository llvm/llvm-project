//===-- Unittests for floorl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FloorTest.h"

#include "src/math/floorl.h"

LIST_FLOOR_TESTS(long double, __llvm_libc::floorl)
