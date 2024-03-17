//===-- Unittests for nextdownl -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NextDownTest.h"

#include "src/math/nextdownl.h"

LIST_NEXTDOWN_TESTS(long double, LIBC_NAMESPACE::nextdownl)
