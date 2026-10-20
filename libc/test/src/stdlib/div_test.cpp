//===-- Unittests for div -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DivTest.h"

#include "hdr/types/div_t.h"
#include "src/stdlib/div.h"

LIST_DIV_TESTS(int, div_t, LIBC_NAMESPACE::div)
