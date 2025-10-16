//===-- Unittests for fminimum_magl----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMinimumMagTest.h"

#include "src/math/fminimum_magl.h"

LIST_FMINIMUM_MAG_TESTS(long double, LIBC_NAMESPACE::fminimum_magl)
