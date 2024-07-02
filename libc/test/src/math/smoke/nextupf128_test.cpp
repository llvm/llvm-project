//===-- Unittests for nextupf128 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NextUpTest.h"

#include "src/math/nextupf128.h"

LIST_NEXTUP_TESTS(float128, LIBC_NAMESPACE::nextupf128)
