//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for fmaximumf80.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float80.h"
#include "src/math/fmaximumf80.h"
#include "test/src/math/smoke/FMaximumTest.h"

using LIBC_NAMESPACE::fputil::Float80;

LIST_FMAXIMUM_TESTS(Fmaximumf80, Float80, LIBC_NAMESPACE::fmaximumf80)
