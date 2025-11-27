//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_COMMON
#define _LIBSYCL_COMMON

#include <sycl/__impl/detail/config.hpp>

#include <cstddef>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// Minimal span-like view
template <class T> struct range_view {
  T *ptr{};
  size_t len{};
  T *begin() const { return ptr; }
  T *end() const { return ptr + len; }
  T &operator[](size_t i) const { return ptr[i]; }
  size_t size() const { return len; }
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_COMMON
