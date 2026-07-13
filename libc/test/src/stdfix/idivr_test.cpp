//===-- Unittests for idivr -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IdivFxTest.h"

#include "llvm-libc-macros/stdfix-macros.h" // fract
#include "src/stdfix/idivr.h"

LIST_IDIVFX_TESTS(r, fract, int, LIBC_NAMESPACE::idivr);
