//===- constexpr_hash.h - Classes for representing statements -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a constexpr `hash`
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONSTEXPR_HASH_H
#define SUPPORT_CONSTEXPR_HASH_H

#include <__type_traits/is_unqualified.h>
#include <utility>

#include "test_macros.h"
#include <type_traits>
#include <bit>
#include <cstddef>

namespace support {

#if TEST_STD_VER >= 26

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
  constexpr_hash() noexcept                        = default;
  constexpr_hash& operator=(const constexpr_hash&) = default;

  [[nodiscard]] constexpr _LIBCPP_HIDE_FROM_ABI size_t operator()(std::nullptr_t) const noexcept {
    return 662607004ull;
  }
};

template <typename Key>
  requires requires(Key t) {
    { std::bool_constant<std::__is_unqualified_v<Key>>() } -> std::same_as<std::true_type>;

    {
      std::bool_constant< std::bool_constant<std::is_integral_v<Key>>() ||
                          std::bool_constant<std::is_floating_point_v<Key>>() ||
                          std::bool_constant<std::is_enum_v<Key>>() >()
    } -> std::same_as<std::true_type>;
  }
struct constexpr_hash<Key> {
  [[nodiscard]] constexpr _LIBCPP_HIDE_FROM_ABI size_t operator()(const Key& __v) const noexcept {
    if constexpr (std::is_integral_v<Key>) {
      if constexpr (sizeof(Key) <= sizeof(size_t)) {
        return static_cast<size_t>(__v);
      } else {
        constexpr size_t multiple           = sizeof(Key) / sizeof(size_t);
        std::array<size_t, multiple> region = std::bit_cast<std::array<size_t, multiple>>(__v);

        // TODO: 0, 1, 2, 3, 4
        return region[multiple - 1]; // TODO: hash-ing
      }
    } else if constexpr (std::is_floating_point_v<Key>) {
      if (__v == 0.0f)

        return 0;
      // return __scalar_hash<Key>::operator()(__v);

      constexpr size_t multiple         = sizeof(Key) / sizeof(char);
      std::array<char, multiple> region = std::bit_cast<std::array<char, multiple>>(__v);

      // TODO: pack indexing?
      size_t res = 0;
      for (size_t i = 0; i < multiple; ++i) {
        res ^= region[i];
      }
      return res;

      // TODO: 0, 1, 2, 3, 4
      // return region[multiple - 1]; // TODO: hash-ing
      // return 1234ULL;

    } else if constexpr (std::is_enum_v<Key>) {
      using type = decltype(std::to_underlying(std::declval<Key>()));
      return constexpr_hash<type>()(static_cast<type>(__v));
    }
    __builtin_unreachable(); // todo: revisit
  }

  constexpr_hash() noexcept                        = default;
  constexpr_hash& operator=(const constexpr_hash&) = default;
};

} // namespace support

#endif

#endif // SUPPORT_CONSTEXPR_HASH_H
