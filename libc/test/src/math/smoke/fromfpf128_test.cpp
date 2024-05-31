//===-- Unittests for fromfpf128 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FromfpTest.h"

#include "src/math/fromfpf128.h"

LIST_FROMFP_TESTS(float128, LIBC_NAMESPACE::fromfpf128)
