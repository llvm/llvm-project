//===-- Internal utils for wctype conversion code - zip ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helper iterator for zipping two containers together. `__support` code can not
// have dependencies, so we implement a minimal zip here used internally.

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_ZIP_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_ZIP_H

#include "src/__support/CPP/tuple.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace conversion_utils {

namespace {

template <typename It1, typename It2> class ZipIterator {
public:
  using value_type = cpp::tuple<It1, It2>;

  LIBC_INLINE constexpr ZipIterator(It1 it1, It2 it2) : it1(it1), it2(it2) {}

  LIBC_INLINE constexpr const ZipIterator &operator++() const {
    ++it1;
    ++it2;
    return *this;
  }

  LIBC_INLINE constexpr bool operator!=(const ZipIterator &other) const {
    return it1 != other.it1 && it2 != other.it2;
  }

  LIBC_INLINE constexpr auto operator*() const { return cpp::tie(*it1, *it2); }

private:
  mutable It1 it1;
  mutable It2 it2;
};

template <typename Container1, typename Container2> class ZipWrapper {
public:
  LIBC_INLINE constexpr ZipWrapper(Container1 &c1, Container2 &c2)
      : c1(c1), c2(c2) {}

  LIBC_INLINE constexpr auto begin() const {
    return ZipIterator(c1.begin(), c2.begin());
  }
  LIBC_INLINE constexpr auto end() const {
    return ZipIterator(c1.end(), c2.end());
  }

private:
  Container1 &c1;
  Container2 &c2;
};

} // namespace

// Helper `zip` function
template <typename C1, typename C2>
LIBC_INLINE static constexpr ZipWrapper<C1, C2> zip(C1 &c1, C2 &c2) {
  return ZipWrapper<C1, C2>(c1, c2);
}

} // namespace conversion_utils

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_UTILS_ZIP_H
