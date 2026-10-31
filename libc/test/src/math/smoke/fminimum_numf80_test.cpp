//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for fminimum_numf80.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float80.h"
#include "src/math/fminimum_numf80.h"
#include "test/src/math/smoke/FMinimumNumTest.h"

using LIBC_NAMESPACE::fputil::Float80;

LIST_FMINIMUM_NUM_TESTS(Fminimum_numf80, Float80,
                        LIBC_NAMESPACE::fminimum_numf80)
