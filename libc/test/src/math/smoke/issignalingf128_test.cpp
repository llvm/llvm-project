//===-- Unittests for issignalingf128 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IsSignalingTest.h"

#include "src/math/issignalingf128.h"

LIST_ISSIGNALING_TESTS(float128, LIBC_NAMESPACE::issignalingf128)
