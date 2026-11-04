//===-- Unittests for llogbf80 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/__support/FPUtil/float80.h"
#include "src/math/llogbf80.h"

using LIBC_NAMESPACE::fputil::Float80;

LIST_INTLOGB_TESTS(Llogbf80, long, Float80, LIBC_NAMESPACE::llogbf80);
