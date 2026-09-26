//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-random-device

// Ensure layout of std::random_device is compatible.

#include <random>

#include "test_macros.h"

#if defined(_LIBCPP_USING_DEV_RANDOM) ||                                                                               \
    (!defined(_LIBCPP_ABI_NO_RANDOM_DEVICE_COMPATIBILITY_LAYOUT) && (defined(__APPLE__) || defined(__GLIBC__)))

static_assert(sizeof(std::random_device) == sizeof(int), "");
static_assert(TEST_ALIGNOF(std::random_device) == TEST_ALIGNOF(int), "");

#else

static_assert(sizeof(std::random_device) == 1, "");
static_assert(TEST_ALIGNOF(std::random_device) == 1, "");

#endif
