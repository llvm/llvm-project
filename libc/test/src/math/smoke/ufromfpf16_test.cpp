//===-- Unittests for ufromfpf16 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UfromfpTest.h"

#include "src/math/ufromfpf16.h"

LIST_UFROMFP_TESTS(float16, LIBC_NAMESPACE::ufromfpf16)
