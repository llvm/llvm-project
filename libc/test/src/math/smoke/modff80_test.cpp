//===-- Unittests for modff80 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModfTest.h"

#include "src/math/modff80.h"

LIST_MODF_TESTS(Modff80, Float80, LIBC_NAMESPACE::modff80)
