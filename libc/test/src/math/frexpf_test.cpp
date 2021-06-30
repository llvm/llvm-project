//===-- Unittests for frexpf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FrexpTest.h"

#include "src/math/frexpf.h"

LIST_FREXP_TESTS(float, __llvm_libc::frexpf)
