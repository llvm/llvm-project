//===-- Unittests for fminimum_numf----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMinimumNumTest.h"

#include "src/math/fminimum_numf.h"

LIST_FMINIMUM_NUM_TESTS(float, LIBC_NAMESPACE::fminimum_numf)
