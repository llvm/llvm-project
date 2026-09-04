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

// Check the ABI of std::locale::id

#include <cstdint>
#include <locale>
#include <type_traits>

static_assert(sizeof(std::locale::id) == 2 * sizeof(std::uintptr_t), "");
static_assert(alignof(std::locale::id) == alignof(std::uintptr_t), "");
static_assert(std::is_trivially_destructible<std::locale::id>::value, "");

struct IDLayout {
  IDLayout() {}
  uintptr_t a;
  int32_t b;
};

static_assert(std::__datasizeof_v<std::locale::id> == std::__datasizeof_v<IDLayout>, "");
