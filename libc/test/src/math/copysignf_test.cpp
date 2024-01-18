//===-- Unittests for copysignf -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CopySignTest.h"

#include "src/math/copysignf.h"

LIST_COPYSIGN_TESTS(float, LIBC_NAMESPACE::copysignf)
