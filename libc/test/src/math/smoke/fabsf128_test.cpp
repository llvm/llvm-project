//===-- Unittests for fabsf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FAbsTest.h"

#include "src/math/fabsf128.h"

LIST_FABS_TESTS(float128, LIBC_NAMESPACE::fabsf128)
