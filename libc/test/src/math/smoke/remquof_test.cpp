//===-- Unittests for remquof ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RemQuoTest.h"

#include "src/math/remquof.h"

LIST_REMQUO_TESTS(float, LIBC_NAMESPACE::remquof)
