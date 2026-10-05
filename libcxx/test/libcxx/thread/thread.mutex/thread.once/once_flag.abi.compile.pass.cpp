//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads, c++03

// <mutex>

// Check the ABI of std::once_flag

#include <mutex>
#include <type_traits>

static_assert(sizeof(std::once_flag) == sizeof(void*), "");
static_assert(alignof(std::once_flag) == alignof(void*), "");
static_assert(std::is_trivially_destructible<std::once_flag>::value, "");
