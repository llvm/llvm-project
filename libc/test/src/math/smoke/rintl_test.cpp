//===-- Unittests for rintl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RIntTest.h"

#include "src/math/rintl.h"

LIST_RINT_TESTS(long double, LIBC_NAMESPACE::rintl)
