//===-- Unittests for nexttowardbf16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NextTowardTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/nexttowardbf16.h"

LIST_NEXTTOWARD_TESTS(bfloat16, LIBC_NAMESPACE::nexttowardbf16)
