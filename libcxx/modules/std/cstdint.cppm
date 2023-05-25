// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <cstdint>

export module std:cstdint;
export namespace std {
  // signed
  using std::int8_t _LIBCPP_USING_IF_EXISTS;
  using std::int16_t _LIBCPP_USING_IF_EXISTS;
  using std::int32_t _LIBCPP_USING_IF_EXISTS;
  using std::int64_t _LIBCPP_USING_IF_EXISTS;

  using std::int_fast16_t;
  using std::int_fast32_t;
  using std::int_fast64_t;
  using std::int_fast8_t;

  using std::int_least16_t;
  using std::int_least32_t;
  using std::int_least64_t;
  using std::int_least8_t;

  using std::intmax_t;

  using std::intptr_t _LIBCPP_USING_IF_EXISTS;

  // unsigned
  using std::uint8_t _LIBCPP_USING_IF_EXISTS;
  using std::uint16_t _LIBCPP_USING_IF_EXISTS;
  using std::uint32_t _LIBCPP_USING_IF_EXISTS;
  using std::uint64_t _LIBCPP_USING_IF_EXISTS;

  using std::uint_fast16_t;
  using std::uint_fast32_t;
  using std::uint_fast64_t;
  using std::uint_fast8_t;

  using std::uint_least16_t;
  using std::uint_least32_t;
  using std::uint_least64_t;
  using std::uint_least8_t;

  using std::uintmax_t;

  using std::uintptr_t _LIBCPP_USING_IF_EXISTS;
} // namespace std
