//===-- Unittests for fdiv ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DivTest.h"

#include "src/math/fdiv.h"

LIST_DIV_TESTS(float, double, LIBC_NAMESPACE::fdiv)
