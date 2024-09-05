//===-- Unittests for f16divl ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DivTest.h"

#include "src/math/f16divl.h"

LIST_DIV_TESTS(float16, long double, LIBC_NAMESPACE::f16divl)
