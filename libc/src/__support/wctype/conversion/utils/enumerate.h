//===-- Internal utils for wctype conversion code - enumerate ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_ENUMERATE_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_ENUMERATE_H

#include "hdr/types/size_t.h"
#include "src/__support/CPP/tuple.h"
#include "src/__support/CPP/utility/forward.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace conversion_utils {

namespace {

template <typename Iterable> struct Enumerate {
  Iterable iterable;

  LIBC_INLINE constexpr Enumerate(Iterable &&iter)
      : iterable(cpp::forward<Iterable>(iter)) {}

  struct Iterator {
    size_t index;
    decltype(iterable.begin()) it;

    LIBC_INLINE constexpr auto operator*() const {
      return cpp::tuple<size_t, decltype(*it)>(index, *it);
    }

    LIBC_INLINE constexpr Iterator &operator++() {
      ++index;
      ++it;
      return *this;
    }

    LIBC_INLINE constexpr bool operator!=(const Iterator &other) const {
      return it != other.it;
    }
  };

  LIBC_INLINE constexpr Iterator begin() const { return {0, iterable.begin()}; }

  LIBC_INLINE constexpr Iterator end() const { return {0, iterable.end()}; }
};

} // namespace

template <typename Iterable>
LIBC_INLINE static constexpr Enumerate<Iterable>
enumerate(Iterable &&iterable) {
  return Enumerate<Iterable>(cpp::forward<Iterable>(iterable));
}

} // namespace conversion_utils

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_ENUMERATE_H
