//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// template <class T> class optional::iterator;
// template <class T> class optional::const_iterator;

#include <optional>

template <typename T>
concept has_iterator_aliases = requires {
  typename T::iterator;
  typename T::const_iterator;
};

static_assert(has_iterator_aliases<std::optional<int>>);
static_assert(has_iterator_aliases<std::optional<const int>>);

// TODO: Uncomment these once P2988R12 is implemented, as they would be testing optional<T&>

// static_assert(!has_iterator_aliases<std::optional<int (&)[]>>);
// static_assert(!has_iterator_aliases<std::optional<void (&)(int, char)>>);
