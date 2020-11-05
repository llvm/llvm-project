//===-- Unittests for ldexp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LdExpTest.h"

#include "include/math.h"
#include "src/math/ldexp.h"
#include "utils/CPP/Functional.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"

#include <limits.h>

LIST_LDEXP_TESTS(double, __llvm_libc::ldexp)
