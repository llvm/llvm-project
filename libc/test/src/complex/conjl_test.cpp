//===-- Unittests for conjl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConjTest.h"

#include "src/complex/conjl.h"

LIST_CONJ_TESTS(_Complex long double, long double, LIBC_NAMESPACE::conjl)
