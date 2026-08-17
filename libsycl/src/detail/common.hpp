//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of functions commonly used in SYCL
/// runtime implementation.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_COMMON_HPP
#define _LIBSYCL___IMPL_DETAIL_COMMON_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstddef>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

constexpr bool isPowerOf2(std::size_t n) {
  return (n != 0) && ((n & (n - 1)) == 0);
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_COMMON_HPP
