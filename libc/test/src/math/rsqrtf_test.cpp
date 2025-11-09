//===-- Unittests for rsqrtf ----------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RsqrtTest.h"

#include "src/math/rsqrtf.h"

LIST_RSQRT_TESTS(float, LIBC_NAMESPACE::rsqrtf)
