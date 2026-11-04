//===-- Unittests for logbf80 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LogbTest.h"

#include "src/math/logbf80.h"

LIST_LOGB_TESTS(Logbf80, Float80, LIBC_NAMESPACE::logbf80)
