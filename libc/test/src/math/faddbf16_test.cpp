//===-- Unittests for faddbf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AddTest.h"

#include "src/math/faddbf16.h"

// FIXME: this fails
LIST_ADD_TESTS(bfloat16, bfloat16, LIBC_NAMESPACE::faddbf16)
