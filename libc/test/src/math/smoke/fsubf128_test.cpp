//===-- Unittests for fsubf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubTest.h"

#include "src/math/fsubf128.h"

LIST_SUB_TESTS(float, float128, LIBC_NAMESPACE::fsubf128)
