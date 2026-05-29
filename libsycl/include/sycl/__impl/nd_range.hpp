//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 nd_range class.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ND_RANGE_HPP
#define _LIBSYCL___IMPL_ND_RANGE_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/index_space_classes.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class UnifiedRangeView;
} // namespace detail

// SYCL 2020 4.9.1.2. nd_range class.
/// nd_range<int Dimensions> defines the iteration domain of both the
/// work-groups and the overall dispatch.
template <int Dimensions = 1> class nd_range {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "nd_range can only be 1-, 2-, or 3-dimensional.");

public:
  static constexpr int dimensions = Dimensions;

  nd_range(const nd_range<Dimensions> &rhs) = default;
  nd_range(nd_range<Dimensions> &&rhs) = default;
  nd_range<Dimensions> &operator=(const nd_range<Dimensions> &rhs) = default;
  nd_range<Dimensions> &operator=(nd_range<Dimensions> &&rhs) = default;

  friend bool operator==(const nd_range<Dimensions> &lhs,
                         const nd_range<Dimensions> &rhs) {
    return (rhs.MGlobalSize == lhs.MGlobalSize) &&
           (rhs.MLocalSize == lhs.MLocalSize) && (rhs.MOffset == lhs.MOffset);
  }

  friend bool operator!=(const nd_range<Dimensions> &lhs,
                         const nd_range<Dimensions> &rhs) {
    return !(lhs == rhs);
  }

  __SYCL2020_DEPRECATED("offset is deprecated in SYCL2020")
  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize,
           id<Dimensions> offset) noexcept
      : MGlobalSize(globalSize), MLocalSize(localSize), MOffset(offset) {}

  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize)
      : MGlobalSize(globalSize), MLocalSize(localSize),
        MOffset(id<Dimensions>()) {}

  /// \return the constituent global range.
  range<Dimensions> get_global_range() const noexcept { return MGlobalSize; }

  /// \return the constituent local range.
  range<Dimensions> get_local_range() const noexcept { return MLocalSize; }

  /// This range would result from globalSize/localSize as provided on
  /// construction.
  /// \return a range representing the number of groups in each dimension.
  range<Dimensions> get_group_range() const noexcept {
    return MGlobalSize / MLocalSize;
  }

  /// Deprecated in SYCL 2020.
  /// \return the constituent offset.
  __SYCL2020_DEPRECATED("offset is deprecated in SYCL2020")
  id<Dimensions> get_offset() const noexcept { return MOffset; }

protected:
  range<Dimensions> MGlobalSize;
  range<Dimensions> MLocalSize;
  id<Dimensions> MOffset;

  friend class detail::UnifiedRangeView;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ND_RANGE_HPP
