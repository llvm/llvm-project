//===-- Unittests for floorf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FloorTest.h"

#include "src/math/floorf16.h"

LIST_FLOOR_TESTS(float16, LIBC_NAMESPACE::floorf16)
