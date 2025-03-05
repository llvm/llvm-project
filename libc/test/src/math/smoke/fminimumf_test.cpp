//===-- Unittests for fminimumf--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMinimumTest.h"

#include "src/math/fminimumf.h"

LIST_FMINIMUM_TESTS(float, LIBC_NAMESPACE::fminimumf)
