//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 nd_item class.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ND_ITEM_HPP
#define _LIBSYCL___IMPL_ND_ITEM_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/linearization.hpp>
#include <sycl/__impl/group.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/nd_range.hpp>
#include <sycl/__impl/sub_group.hpp>
#include <sycl/__spirv/spirv_vars.hpp>

#include <cstddef>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class Builder;
} // namespace detail

// SYCL2020 4.9.1.5. nd_item class.
/// nd_item<int Dimensions> identifies an instance of the function object
/// executing at each point in an nd_range<int Dimensions> passed to a
/// parallel_for call.
template <int Dimensions = 1> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

  nd_item(const nd_item &rhs) = default;
  nd_item(nd_item &&rhs) = default;
  nd_item &operator=(const nd_item &rhs) = default;
  nd_item &operator=(nd_item &&rhs) = default;

  friend bool operator==(const nd_item &, const nd_item &) {
    // https://github.com/KhronosGroup/SYCL-Docs/issues/532
    return true;
  }

  friend bool operator!=(const nd_item &lhs, const nd_item &rhs) {
    return !(lhs == rhs);
  }

  /// \return the constituent global id representing the work-item’s position in
  /// the global iteration space.
  id<Dimensions> get_global_id() const noexcept {
    return __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>();
  }

  /// \return the constituent element of the global id representing the
  /// work-item’s position in the nd-range in the given Dimension.
  std::size_t get_global_id(int dimension) const noexcept {
    return get_global_id()[dimension];
  }

  /// \return the constituent global id as a linear index value, representing
  /// the work-item’s position in the global iteration space.
  std::size_t get_global_linear_id() const noexcept {
    id<Dimensions> adjustedIndex = get_global_id();
    const id<Dimensions> offset =
        __spirv::initBuiltInGlobalOffset<Dimensions, id<Dimensions>>();
    if constexpr (Dimensions == 1) {
      adjustedIndex[0] -= offset[0];
    } else if constexpr (Dimensions == 2) {
      adjustedIndex[0] -= offset[0];
      adjustedIndex[1] -= offset[1];
    } else {
      adjustedIndex[0] -= offset[0];
      adjustedIndex[1] -= offset[1];
      adjustedIndex[2] -= offset[2];
    }
    return detail::linearize_id(adjustedIndex, get_global_range());
  }

  /// \return the constituent local id representing the work-item’s position
  /// within the current work-group.
  id<Dimensions> get_local_id() const noexcept {
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
  }

  /// \return the constituent element of the local id representing the
  /// work-item’s position within the current work-group in the given Dimension.
  std::size_t get_local_id(int dimension) const noexcept {
    return get_local_id()[dimension];
  }

  /// \return the constituent local id as a linear index value, representing the
  /// work-item’s position within the current work-group.
  std::size_t get_local_linear_id() const noexcept {
    return detail::linearize_id(get_local_id(), get_local_range());
  }

  /// \return the constituent work-group, group representing the work-group's
  /// position within the overall nd-range.
  group<Dimensions> get_group() const noexcept { return group<Dimensions>(); }

  /// \return a sub_group representing the sub-group to which the work-item
  /// belongs.
  sub_group get_sub_group() const noexcept { return sub_group(); }

  /// \return the constituent element of the group id representing the
  /// work-group’s position within the overall nd_range in the given Dimension.
  std::size_t get_group(int dimension) const noexcept {
    return get_group_id()[dimension];
  }

  /// \return the group id as a linear index value.
  std::size_t get_group_linear_id() const noexcept {
    return detail::linearize_id(get_group_id(), get_group_range());
  }

  /// \return the number of work-groups in the iteration space.
  range<Dimensions> get_group_range() const noexcept {
    return __spirv::initBuiltInNumWorkgroups<Dimensions, range<Dimensions>>();
  }

  /// \return the number of work-groups for Dimension in the iteration space.
  std::size_t get_group_range(int dimension) const noexcept {
    return get_group_range()[dimension];
  }

  /// \return a range representing the dimensions of the global iteration space.
  range<Dimensions> get_global_range() const noexcept {
    return __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>();
  }

  /// Equivalent to return get_global_range().get(dimension).
  std::size_t get_global_range(int dimension) const noexcept {
    return get_global_range()[dimension];
  }

  /// \return a range representing the dimensions of the current work-group.
  range<Dimensions> get_local_range() const noexcept {
    return __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>();
  }

  /// Equivalent to return get_local_range().get(dimension).
  std::size_t get_local_range(int dimension) const noexcept {
    return get_local_range()[dimension];
  }

  /// Deprecated in SYCL 2020.
  /// \return an id representing the n-dimensional offset provided to the
  /// constructor of the nd_range and that is added by the runtime to the global
  /// id of each work-item.
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  id<Dimensions> get_offset() const noexcept {
    return __spirv::initBuiltInGlobalOffset<Dimensions, id<Dimensions>>();
  }

  /// \return the nd_range of the current execution.
  nd_range<Dimensions> get_nd_range() const noexcept {
    return nd_range<Dimensions>(
        get_global_range(), get_local_range(),
        __spirv::initBuiltInGlobalOffset<Dimensions, id<Dimensions>>());
  }

  // TODO: add wait_for and async_work_group_copy once builtins are implemented.

protected:
  friend class detail::Builder;

  nd_item() = default;

  id<Dimensions> get_group_id() const {
    return __spirv::initBuiltInWorkgroupId<Dimensions, id<Dimensions>>();
  }
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ND_ITEM_HPP
