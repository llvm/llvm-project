//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helpers for info descriptors.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INFO_DESC_BASE_HPP
#define _LIBSYCL___IMPL_INFO_DESC_BASE_HPP

#include <sycl/__impl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

template <typename Desc, typename DescOf> struct info_desc_tag {};

template <typename Desc, typename DescOf, typename = void>
struct is_info_desc : std::false_type {};

template <typename Desc, typename DescOf>
struct is_info_desc<
    Desc, DescOf,
    std::enable_if_t<std::is_base_of_v<info_desc_tag<Desc, DescOf>, Desc>>>
    : std::true_type {
  using return_type = typename Desc::return_type;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_DESC_BASE_HPP
