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

#include <utility>

#include "test_macros.h"
#include <type_traits>
#include <concepts>
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
concept UnqualifiedArithmeticOrEnum =
    std::is_same_v<Key, std::remove_cvref_t<Key>> &&
    (std::integral<Key> || std::floating_point<Key> || std::is_enum_v<Key>);

template <UnqualifiedArithmeticOrEnum Key>
struct constexpr_hash<Key> {
  [[nodiscard]] constexpr _LIBCPP_HIDE_FROM_ABI size_t operator()(const Key& key) const noexcept {
    if constexpr (std::integral<Key>) {
      if constexpr (sizeof(Key) <= sizeof(size_t)) {
        return static_cast<size_t>(key);
      } else {
        constexpr size_t multiple           = sizeof(Key) / sizeof(size_t);
        std::array<size_t, multiple> region = std::bit_cast<std::array<size_t, multiple>>(key);

        // TODO: 0, 1, 2, 3, 4
        return region[multiple - 1]; // TODO: hash-ing
      }
    } else if constexpr (std::floating_point<Key>) {
      if (key == 0.0f)

        return 0;
      // return __scalar_hash<Key>::operator()(key);

      constexpr size_t multiple         = sizeof(Key) / sizeof(char);
      std::array<char, multiple> region = std::bit_cast<std::array<char, multiple>>(key);

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
      return constexpr_hash<type>()(static_cast<type>(key));
    }
    std::unreachable(); // todo: revisit
  }

  constexpr_hash() noexcept                        = default;
  constexpr_hash& operator=(const constexpr_hash&) = default;
};

} // namespace support

#endif

#endif // SUPPORT_CONSTEXPR_HASH_H
