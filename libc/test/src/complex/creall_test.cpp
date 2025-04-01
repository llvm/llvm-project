//===-- Unittests for creall ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CRealTest.h"

#include "src/complex/creall.h"

LIST_CREAL_TESTS(_Complex long double, long double, LIBC_NAMESPACE::creall)
