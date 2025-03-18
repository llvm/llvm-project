//===-- Unittests for faddf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AddTest.h"

#include "src/math/faddf128.h"

LIST_ADD_TESTS(float, float128, LIBC_NAMESPACE::faddf128)
