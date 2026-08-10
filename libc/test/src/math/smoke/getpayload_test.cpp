//===-- Unittests for getpayload ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GetPayloadTest.h"

#include "src/math/getpayload.h"

LIST_GETPAYLOAD_TESTS(double, LIBC_NAMESPACE::getpayload)
