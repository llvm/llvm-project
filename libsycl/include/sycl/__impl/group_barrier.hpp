//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the SYCL 2020 group_barrier function (4.17.2.3.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_GROUP_BARRIER_HPP
#define _LIBSYCL___IMPL_GROUP_BARRIER_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/group.hpp>
#include <sycl/__impl/memory_enums.hpp>
#include <sycl/__impl/sub_group.hpp>
#include <sycl/__spirv/spirv_types.hpp>

#include <type_traits>

void __spirv_ControlBarrier(std::uint32_t Execution, std::uint32_t Memory,
                            std::uint32_t Semantics);

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// 4.17.2.1. Group type trait.

template <class T> struct is_group : std::false_type {};

template <int Dimensions>
struct is_group<group<Dimensions>> : std::true_type {};

template <> struct is_group<sub_group> : std::true_type {};

template <class T>
inline constexpr bool is_group_v = is_group<std::decay_t<T>>::value;

namespace detail {

static constexpr __spirv::Scope getScope(memory_scope Scope) {
  switch (Scope) {
  case memory_scope::work_item:
    return __spirv::Scope::Invocation;
  case memory_scope::sub_group:
    return __spirv::Scope::Subgroup;
  case memory_scope::work_group:
    return __spirv::Scope::Workgroup;
  case memory_scope::device:
    return __spirv::Scope::Device;
  case memory_scope::system:
    return __spirv::Scope::CrossDevice;
  }
}

template <typename Group> struct group_scope {};

template <int Dimensions> struct group_scope<group<Dimensions>> {
  static constexpr __spirv::Scope value = __spirv::Scope::Workgroup;
};

template <> struct group_scope<::sycl::sub_group> {
  static constexpr __spirv::Scope value = __spirv::Scope::Subgroup;
};

} // namespace detail

/// Blocks until all work-items in group g have reached this synchronization
/// point.
template <typename Group>
std::enable_if_t<is_group_v<Group>>
group_barrier(Group /*G*/, memory_scope FenceScope = Group::fence_scope) {
  __spirv_ControlBarrier(detail::group_scope<Group>::value,
                         detail::getScope(FenceScope),
                         __spirv::MemorySemanticsMask::SequentiallyConsistent |
                             __spirv::MemorySemanticsMask::SubgroupMemory |
                             __spirv::MemorySemanticsMask::WorkgroupMemory);
}

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_GROUP_BARRIER_HPP
