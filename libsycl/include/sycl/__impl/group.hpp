//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 group class.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_GROUP_HPP
#define _LIBSYCL___IMPL_GROUP_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/linearization.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/memory_enums.hpp>
#include <sycl/__spirv/spirv_vars.hpp>

#include <cstddef>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

template <int> class nd_item;

// SYCL2020 4.9.1.7. group class.
/// The group class encapsulates all functionality required to represent a
/// particular work-group within a parallel execution.
template <int Dimensions = 1> class group {
public:
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = std::size_t;
  static constexpr int dimensions = Dimensions;
  static constexpr memory_scope fence_scope = memory_scope::work_group;

  group(const group &rhs) = default;
  group(group &&rhs) = default;
  group &operator=(const group &rhs) = default;
  group &operator=(group &&rhs) = default;

  /// \return an id representing the index of the work-group within the global
  /// nd-range for every dimension.
  id<Dimensions> get_group_id() const noexcept {
    return __spirv::initBuiltInWorkgroupId<Dimensions, id<Dimensions>>();
  }

  /// Equivalent to `return get_group_id()[dimension]`.
  std::size_t get_group_id(int dimension) const noexcept {
    return get_group_id()[dimension];
  }

  /// \return a SYCL id representing the calling work-item’s position within the
  /// work-group.
  id<Dimensions> get_local_id() const noexcept {
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
  }

  /// Equivalent to `return get_local_id()[dimension]`.
  std::size_t get_local_id(int dimension) const noexcept {
    return get_local_id()[dimension];
  }

  /// \return a SYCL range representing all dimensions of the local range.
  range<Dimensions> get_local_range() const noexcept {
    return __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>();
  }

  /// Equivalent to `return get_local_range()[dimension]`.
  std::size_t get_local_range(int dimension) const noexcept {
    return get_local_range()[dimension];
  }

  /// \return a SYCL range representing the number of work-groups in the
  /// nd-range.
  range<Dimensions> get_group_range() const noexcept {
    return __spirv::initBuiltInNumWorkgroups<Dimensions, range<Dimensions>>();
  }

  /// Equivalent to `return get_group_range()[dimension]`.
  std::size_t get_group_range(int dimension) const noexcept {
    return get_group_range()[dimension];
  }

  /// \return a SYCL range representing the maximum number of work-items in any
  /// work-group in the nd-range.
  range<Dimensions> get_max_local_range() const noexcept {
    return get_local_range();
  }

  /// Equivalent to `return get_group_id(dimension)`.
  std::size_t operator[](int dimension) const noexcept {
    return get_group_id(dimension);
  }

  /// \return the linearized work-group id within the nd-range.
  std::size_t get_group_linear_id() const noexcept {
    return detail::linearize_id(get_group_id(), get_group_range());
  }

  /// \return a linearized version of the calling work-item’s local id.
  std::size_t get_local_linear_id() const noexcept {
    return detail::linearize_id(get_local_id(), get_local_range());
  }

  /// \return the total number of work-groups in the nd-range.
  std::size_t get_group_linear_range() const noexcept {
    auto groupRange = get_group_range();
    return multiply_all_dims(groupRange);
  }

  /// \return the total number of work-items in this work-group.
  std::size_t get_local_linear_range() const noexcept {
    auto localRange = get_local_range();
    return multiply_all_dims(localRange);
  }

  /// \return true for exactly one work-item in the work-group, if the calling
  /// work-item is the leader of the work-group, and false for all other
  /// work-items in the work-group.
  bool leader() const noexcept { return (get_local_linear_id() == 0); }

  // TODO: implement parallel_for_work_item, async_work_group_copy and wait_for.

protected:
  group() = default;

  static std::size_t
  multiply_all_dims(const range<Dimensions> &Range) noexcept {
    if constexpr (Dimensions == 1) {
      return Range[0];
    } else if constexpr (Dimensions == 2) {
      return Range[0] * Range[1];
    } else {
      return Range[0] * Range[1] * Range[2];
    }
  }

  friend class sycl::nd_item<Dimensions>;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_GROUP_HPP
