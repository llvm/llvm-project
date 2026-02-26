//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INFO_DEVICE_TYPE_HPP
#define _LIBSYCL___IMPL_INFO_DEVICE_TYPE_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstdint>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace info {

// SYCL 2020 4.6.4.7.1. Device type.
enum class device_type : std::uint32_t {
  cpu = 0,
  gpu,
  accelerator,
  custom,
  automatic,
  host, // Deprecated by SYCL 2020
  all
};

} // namespace info

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_DEVICE_TYPE_HPP
