//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__type_traits/datasizeof.h>
#include <cstdint>

static_assert(std::__datasizeof_v<std::int8_t> == 1, "");
static_assert(std::__datasizeof_v<std::int16_t> == 2, "");
static_assert(std::__datasizeof_v<std::int32_t> == 4, "");
static_assert(std::__datasizeof_v<std::int64_t> == 8, "");

struct OneBytePadding {
  OneBytePadding() {}

  std::int16_t a;
  std::int8_t b;
};

#if defined(_WIN32) && !defined(__MINGW32__)
static_assert(std::__datasizeof_v<OneBytePadding> == 4, "");
#else
static_assert(std::__datasizeof_v<OneBytePadding> == 3, "");
#endif

struct InBetweenPadding {
  InBetweenPadding() {}

  std::int32_t a;
  std::int8_t b;
  std::int16_t c;
};

static_assert(std::__datasizeof_v<InBetweenPadding> == 8, "");
