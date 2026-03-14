//===-- Unittests for ceill -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CeilTest.h"

#include "src/math/ceill.h"

LIST_CEIL_TESTS(long double, LIBC_NAMESPACE::ceill)
