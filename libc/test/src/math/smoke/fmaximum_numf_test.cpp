//===-- Unittests for fmaximum_numf----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMaximumNumTest.h"

#include "src/math/fmaximum_numf.h"

LIST_FMAXIMUM_NUM_TESTS(float, LIBC_NAMESPACE::fmaximum_numf)
