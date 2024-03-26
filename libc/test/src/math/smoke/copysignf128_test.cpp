//===-- Unittests for copysignf128 ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CopySignTest.h"

#include "src/math/copysignf128.h"

LIST_COPYSIGN_TESTS(float128, LIBC_NAMESPACE::copysignf128)
