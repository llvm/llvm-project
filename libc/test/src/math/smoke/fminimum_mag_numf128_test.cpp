//===-- Unittests for fminimum_mag_numf128---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMinimumMagNumTest.h"

#include "src/math/fminimum_mag_numf128.h"

LIST_FMINIMUM_MAG_NUM_TESTS(float128, LIBC_NAMESPACE::fminimum_mag_numf128)
