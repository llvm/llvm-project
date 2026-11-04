//===-- Unittests for ilogbf80 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/math/ilogbf80.h"

LIST_INTLOGB_TESTS(Ilogbf80, int, Float80, LIBC_NAMESPACE::ilogbf80);
