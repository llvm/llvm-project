// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_ALGORITHMS_H
#define TEST_SUPPORT_ALGORITHMS_H

#include <cstddef>
#include <memory>
#include <type_traits>

#include "test_macros.h"

// This file intends to provide algorithms backported from newer standard revisions.

namespace util {

template <class T, typename std::enable_if<!std::is_array<T>::value, int>::type = 0>
TEST_CONSTEXPR_CXX20 void destroy_at(T* p) {
  p->~T();
}
template <class T, typename std::enable_if<std::is_array<T>::value, int>::type = 0>
TEST_CONSTEXPR_CXX20 void destroy_at(T* p) {
  static_assert(std::extent<T>::value > 0, "must destroy a bounded array");
  for (std::size_t i = 0; i != std::extent<T>::value; ++i)
    util::destroy_at(i + *p);
}

template <class Iterator>
TEST_CONSTEXPR_CXX20 void destroy(Iterator first, Iterator last) {
  for (; first != last; ++first)
    util::destroy_at(std::addressof(*first));
}

template <class Iterator, class Size>
TEST_CONSTEXPR_CXX20 Iterator destroy_n(Iterator first, Size n) {
  for (; n > 0; ++first, (void)--n)
    util::destroy_at(std::addressof(*first));
  return first;
}

} // namespace util

#endif // TEST_SUPPORT_ALGORITHMS_H
