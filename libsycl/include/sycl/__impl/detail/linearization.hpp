//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helpers for SYCL index/range linearization. Follows SYCL2020 3.11.1.
/// Linearization.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_LINEARIZATION_HPP
#define _LIBSYCL___IMPL_DETAIL_LINEARIZATION_HPP

#include <sycl/__impl/index_space_classes.hpp>

#include <cstddef>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

template <int Dimensions>
inline std::size_t linearize_id(const id<Dimensions> &Index,
                                const range<Dimensions> &Extent) noexcept {
  if constexpr (Dimensions == 1) {
    return Index[0];
  } else if constexpr (Dimensions == 2) {
    return Index[0] * Extent[1] + Index[1];
  } else {
    return Index[0] * Extent[1] * Extent[2] + Index[1] * Extent[2] + Index[2];
  }
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_LINEARIZATION_HPP
