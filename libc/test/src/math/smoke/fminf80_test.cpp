//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for fminf80.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float80.h"
#include "src/math/fminf80.h"
#include "test/src/math/smoke/FMinTest.h"

using LIBC_NAMESPACE::fputil::Float80;

LIST_FMIN_TESTS(Fminf80, Float80, LIBC_NAMESPACE::fminf80)
