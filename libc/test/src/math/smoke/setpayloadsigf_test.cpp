//===-- Unittests for setpayloadsigf --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SetPayloadSigTest.h"

#include "src/math/setpayloadsigf.h"

LIST_SETPAYLOADSIG_TESTS(float, LIBC_NAMESPACE::setpayloadsigf)
