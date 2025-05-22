//===-- Unittests for nexttowardl -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NextTowardTest.h"

#include "src/math/nexttowardl.h"

LIST_NEXTTOWARD_TESTS(long double, LIBC_NAMESPACE::nexttowardl)
