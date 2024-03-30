//===-- Unittests for fmaximum_magl----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMaximumMagTest.h"

#include "src/math/fmaximum_magl.h"

LIST_FMAXIMUM_MAG_TESTS(long double, LIBC_NAMESPACE::fmaximum_magl)
