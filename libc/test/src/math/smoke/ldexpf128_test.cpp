//===-- Unittests for ldexpf128 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LdExpTest.h"

#include "src/math/ldexpf128.h"

LIST_LDEXP_TESTS(float128, LIBC_NAMESPACE::ldexpf128);
