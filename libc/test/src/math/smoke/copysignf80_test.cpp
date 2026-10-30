//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for copysignf80.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float80.h"
#include "src/math/copysignf80.h"
#include "test/src/math/smoke/CopySignTest.h"

using LIBC_NAMESPACE::fputil::Float80;

LIST_COPYSIGN_TESTS(Copysignf80, Float80, LIBC_NAMESPACE::copysignf80)
