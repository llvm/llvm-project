//===-- Unittests for rdivi -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DivITest.h"

#include "llvm-libc-macros/stdfix-macros.h" // fract
#include "src/stdfix/rdivi.h"

LIST_DIVI_TESTS(r, fract, LIBC_NAMESPACE::rdivi);
