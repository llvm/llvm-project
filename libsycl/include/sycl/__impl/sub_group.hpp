//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 sub_group class.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_SUB_GROUP_HPP
#define _LIBSYCL___IMPL_SUB_GROUP_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/memory_enums.hpp>
#include <sycl/__spirv/spirv_vars.hpp>

#include <cstdint>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

template <int> class nd_item;

// SYCL 2020 4.9.1.8. sub_group class.
/// The sub_group class encapsulates all functionality required to represent a
/// particular sub-group within a parallel execution.
class sub_group {
public:
  using id_type = id<1>;
  using range_type = sycl::range<1>;
  using linear_id_type = std::uint32_t;
  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  sub_group(const sub_group &rhs) = default;
  sub_group(sub_group &&rhs) = default;
  sub_group &operator=(const sub_group &rhs) = default;
  sub_group &operator=(sub_group &&rhs) = default;

  friend bool operator==(const sub_group &lhs, const sub_group &rhs) {
    return lhs.get_group_id() == rhs.get_group_id();
  }

  friend bool operator!=(const sub_group &lhs, const sub_group &rhs) {
    return !(lhs == rhs);
  }

  /// \return an id representing the index of the sub-group within the
  /// work-group.
  id_type get_group_id() const noexcept { return __spirv_BuiltInSubgroupId(); }

  /// \return a SYCL id representing the calling work-item’s position within the
  /// sub-group.
  id_type get_local_id() const noexcept {
    return __spirv_BuiltInSubgroupLocalInvocationId();
  }

  /// \return a range representing the size of the sub-group.
  range_type get_local_range() const noexcept {
    return __spirv_BuiltInSubgroupSize();
  }

  /// \return a range representing the number of sub-groups within the
  /// work-group.
  range_type get_group_range() const noexcept {
    return __spirv_BuiltInNumSubgroups();
  }

  /// \return a range representing the maximum number of work-items permitted in
  /// a sub-group for the executing kernel.
  range_type get_max_local_range() const noexcept {
    return __spirv_BuiltInSubgroupMaxSize();
  }

  /// Equivalent to return get_group_id()[0].
  linear_id_type get_group_linear_id() const noexcept {
    return static_cast<linear_id_type>(get_group_id()[0]);
  }

  /// Equivalent to return get_local_id()[0].
  linear_id_type get_local_linear_id() const noexcept {
    return static_cast<linear_id_type>(get_local_id()[0]);
  }

  /// Equivalent to return get_group_range()[0].
  linear_id_type get_group_linear_range() const noexcept {
    return static_cast<linear_id_type>(get_group_range()[0]);
  }

  /// Equivalent to return get_local_range()[0].
  linear_id_type get_local_linear_range() const noexcept {
    return static_cast<linear_id_type>(get_local_range()[0]);
  }

  /// \return true for exactly one work-item in the sub-group, if the calling
  /// work-item is the leader of the sub-group, and false for all other
  /// work-items in the sub-group.
  bool leader() const noexcept { return get_local_linear_id() == 0; }

protected:
  sub_group() = default;

  template <int dimensions> friend class sycl::nd_item;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_SUB_GROUP_HPP
