//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of SYCL 2020 device info types.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INFO_DEVICE_HPP
#define _LIBSYCL___IMPL_INFO_DEVICE_HPP

#include <sycl/__impl/aspect.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/info/desc_base.hpp>
#include <sycl/__impl/info/device_type.hpp>

#include <cstdint>
#include <string>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class device;
class platform;

namespace detail {
template <typename T>
using is_device_info_desc_t = typename is_info_desc<T, device>::return_type;
} // namespace detail

// SYCL 2020 A.3. Device information descriptors.
namespace info {

enum class partition_property : std::uint32_t {
  no_partition = 0,
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};

enum class partition_affinity_domain : std::uint32_t {
  not_applicable = 0,
  numa,
  L4_cache,
  L3_cache,
  L2_cache,
  L1_cache,
  next_partitionable
};

namespace device {
// SYCL 2020 4.6.4.4. Information descriptors.

struct device_type : detail::info_desc_tag<device_type, sycl::device> {
  using return_type = sycl::info::device_type;
};
struct name : detail::info_desc_tag<name, sycl::device> {
  using return_type = std::string;
};
struct vendor : detail::info_desc_tag<vendor, sycl::device> {
  using return_type = std::string;
};
struct driver_version : detail::info_desc_tag<driver_version, sycl::device> {
  using return_type = std::string;
};
struct platform : detail::info_desc_tag<platform, sycl::device> {
  using return_type = sycl::platform;
};

} // namespace device
} // namespace info

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_DEVICE_HPP
