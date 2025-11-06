//===-- Unittests for frexpf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FrexpTest.h"

#include "src/math/frexpf16.h"

LIST_FREXP_TESTS(float16, LIBC_NAMESPACE::frexpf16);
