//===-- Unittests for carg -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CArgTest.h"

#include "src/complex/carg.h"

LIST_CARG_TESTS(_Complex double, double, LIBC_NAMESPACE::carg)
