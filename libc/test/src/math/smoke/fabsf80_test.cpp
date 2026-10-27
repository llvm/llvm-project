//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for fabsf80.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float80.h"
#include "src/math/fabsf80.h"
#include "test/src/math/smoke/FAbsTest.h"

using LIBC_NAMESPACE::fputil::Float80;

LIST_FABS_TESTS(Fabsf80, Float80, LIBC_NAMESPACE::fabsf80)
