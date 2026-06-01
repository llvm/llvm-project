//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper function class to unify ABI for different kernel
/// ranges.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_UNIFIED_RANGE_VIEW_HPP
#define _LIBSYCL___IMPL_DETAIL_UNIFIED_RANGE_VIEW_HPP

#include <sycl/__impl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// The structure to keep dimension and references to ranges unified for
/// all dimensions.
struct UnifiedRangeView {
  /// Default constructed view matches the single task execution range.
  UnifiedRangeView() = default;
  UnifiedRangeView(const UnifiedRangeView &Desc) = default;
  UnifiedRangeView(UnifiedRangeView &&Desc) = default;
  UnifiedRangeView &operator=(const UnifiedRangeView &Desc) = default;
  UnifiedRangeView &operator=(UnifiedRangeView &&Desc) = default;
  ~UnifiedRangeView() = default;

  // TODO: ctors with sycl::range and nd::range will be added later.

  UnifiedRangeView(const size_t *GlobalSize, const size_t *LocalSize,
                   const size_t *Offset, size_t Dims)
      : MGlobalSize(GlobalSize), MLocalSize(LocalSize), MOffset(Offset),
        MDims(Dims) {}

  const size_t *MGlobalSize = nullptr;
  const size_t *MLocalSize = nullptr;
  const size_t *MOffset = nullptr;
  size_t MDims = 1;
};
} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_UNIFIED_RANGE_VIEW_HPP
