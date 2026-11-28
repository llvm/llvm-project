//===- constexpr_hash.h - A hash function to use during constant evaluation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a constexpr version of `hash` used for testing constexpr for unordered associative containers.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONSTEXPR_HASH_H
#define SUPPORT_CONSTEXPR_HASH_H

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "test_macros.h"

#if TEST_STD_VER >= 26

namespace support {

template <class Key>
struct constexpr_hash {
  constexpr_hash()                                 = delete;
  constexpr_hash(const constexpr_hash&)            = delete;
  constexpr_hash& operator=(const constexpr_hash&) = delete;
};

// we need to disable hashing for `pointer` types, since we can't cast a pointer to a fixed width integer during constant evaluation via std::bit_cast, std::memcpy or union type-aliasing
template <class Key>
struct constexpr_hash<Key*> {
  constexpr_hash()                                 = delete;
  constexpr_hash(const constexpr_hash&)            = delete;
  constexpr_hash& operator=(const constexpr_hash&) = delete;
};

template <>
struct constexpr_hash<std::nullptr_t> {
  [[nodiscard]] constexpr static std::size_t operator()(std::nullptr_t) noexcept {
    return static_cast<std::size_t>(662607004ull);
  }
};

template <typename Key>
concept UnqualifiedArithmeticOrEnum =
    std::is_same_v<Key, std::remove_cvref_t<Key>> &&
    (std::integral<Key> || std::floating_point<Key> || std::is_enum_v<Key>);

template <size_t N>
constexpr std::size_t array_hash(std::array<std::byte, N> arr) noexcept {
  // based on fnv-1
  auto res = static_cast<std::size_t>(14695981039346656037ULL);
  for (auto c : arr) {
    res ^= static_cast<std::size_t>(c);
    res *= static_cast<std::size_t>(1099511628211ULL);
  }
  return res;
}

template <UnqualifiedArithmeticOrEnum Key>
struct constexpr_hash<Key> {
  [[nodiscard]] constexpr static std::size_t operator()(const Key& key) noexcept {
    if constexpr (std::integral<Key>) {
      if constexpr (sizeof(Key) <= sizeof(size_t)) {
        return static_cast<size_t>(key);
      } else {
        return array_hash(std::bit_cast<std::array<std::byte, sizeof(Key)>>(key));
      }
    } else if constexpr (std::floating_point<Key>) {
      if (key == 0.0f)
        return 0;
#  if defined(_WIN32)
      // TODO: Remove this branch once we can add constexpr to std::isnan, std::isinf, and std::signbit on Windows.
      // https://llvm.org/PR214358

      if (__builtin_isnan(key)) {
        return static_cast<std::size_t>(0xc3a5c85c97cb3127ULL);
      }
      if (__builtin_isinf(key)) {
        return __builtin_signbit(key)
                 ? static_cast<std::size_t>(0xb492b66fbe98f273ULL)
                 : static_cast<std::size_t>(0x9ae16a3b2f90404fULL);
      }

#  else
      if (std::isnan(key)) {
        return 0xc3a5c85c97cb3127ULL;
      }
      if (std::isinf(key)) {
        return std::signbit(key) ? 0xb492b66fbe98f273ULL : 0x9ae16a3b2f90404fULL;
      }
#  endif

      // a `long double` is x87 format when mantissa is 64 digits
      constexpr bool is_x87_ld = std::is_same_v<Key, long double> && std::numeric_limits<long double>::digits == 64;

      if constexpr (is_x87_ld) {
        // x87 has padding on last 6 bytes which can be uninitialized and hence, can't be used during constant evaluation
        struct alignas(long double) IntelLongDoubleRep {
          std::array<std::byte, 10> value_bytes;
        };
        auto [value_bytes] = std::bit_cast<IntelLongDoubleRep>(key);

        return array_hash(std::bit_cast<std::array<std::byte, sizeof(value_bytes)>>(value_bytes));
      }

      return array_hash(std::bit_cast<std::array<std::byte, sizeof(Key)>>(key));
    } else if constexpr (std::is_enum_v<Key>) {
      using underlying_type = std::underlying_type_t<Key>;
      return constexpr_hash<underlying_type>()(static_cast<underlying_type>(key));
    } else {
      static_assert(false, "unexpected parameter type");
    }
  }
};

} // namespace support

#endif

#endif // SUPPORT_CONSTEXPR_HASH_H
