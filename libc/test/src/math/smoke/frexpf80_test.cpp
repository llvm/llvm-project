//===-- Unittests for frexpf80 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FrexpTest.h"

#include "src/math/frexpf80.h"

LIST_FREXP_TESTS(Frexpf80, Float80, LIBC_NAMESPACE::frexpf80);
