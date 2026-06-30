//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains SYCL 2020 memory scope enumeration (3.8.3.2.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_MEMORY_ENUMS_HPP
#define _LIBSYCL___IMPL_MEMORY_ENUMS_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstdint>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

enum class memory_scope : std::uint32_t {
  work_item = 0,
  sub_group = 1,
  work_group = 2,
  device = 3,
  system = 4
};

inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;
inline constexpr auto memory_scope_system = memory_scope::system;

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_MEMORY_ENUMS_HPP
