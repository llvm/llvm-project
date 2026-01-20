//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of SYCL 2020 platform info types.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INFO_PLATFORM_HPP
#define _LIBSYCL___IMPL_INFO_PLATFORM_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/info/desc_base.hpp>

#include <string>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class platform;

namespace detail {
template <typename T>
using is_platform_info_desc_t = typename is_info_desc<T, platform>::return_type;
} // namespace detail

// SYCL 2020 A.1. Platform information descriptors.
namespace info {
namespace platform {
// SYCL 2020 4.6.2.4. Information descriptors.
struct version : detail::info_desc_tag<version, sycl::platform> {
  using return_type = std::string;
};
struct name : detail::info_desc_tag<name, sycl::platform> {
  using return_type = std::string;
};
struct vendor : detail::info_desc_tag<vendor, sycl::platform> {
  using return_type = std::string;
};
} // namespace platform
} // namespace info

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_PLATFORM_HPP
