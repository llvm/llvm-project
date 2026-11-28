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
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/info/desc_base.hpp>
#include <sycl/__impl/info/device_type.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class device;
class platform;

namespace detail {
template <typename T>
using is_device_info_desc_t = typename is_info_desc<T, device>::return_type;
template <typename DerivedT, typename ReturnT>
struct DeviceInfoDesc : info_desc_tag<DerivedT, sycl::device> {
  using return_type = ReturnT;
};
} // namespace detail

// SYCL 2020 A.3. Device information descriptors.
namespace info {
// SYCL 2020 4.6.4.7.2. Partition property.
enum class partition_property : std::uint32_t {
  no_partition = 0,
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};

// SYCL 2020 4.6.4.7.3. Partition affinity domain.
enum class partition_affinity_domain : std::uint32_t {
  not_applicable = 0,
  numa,
  L4_cache,
  L3_cache,
  L2_cache,
  L1_cache,
  next_partitionable
};

// SYCL 2020 4.6.4.7.4. Floating point configuration
enum class fp_config : std::uint32_t {
  denorm,
  inf_nan,
  round_to_nearest,
  round_to_zero,
  round_to_inf,
  fma,
  correctly_rounded_divide_sqrt,
  soft_float
};

namespace device {
// SYCL 2020 4.6.4.4. Information descriptors.
#define _LIBSYCL_DEVICE_INFO_DESC(T, ReturnT)                                  \
  struct T : detail::DeviceInfoDesc<T, ReturnT> {}

_LIBSYCL_DEVICE_INFO_DESC(device_type, sycl::info::device_type);
_LIBSYCL_DEVICE_INFO_DESC(vendor_id, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(max_compute_units, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(max_work_item_dimensions, std::uint32_t);
template <int Dimensions = 3> struct max_work_item_sizes;
template <> _LIBSYCL_DEVICE_INFO_DESC(max_work_item_sizes<1>, range<1>);
template <> _LIBSYCL_DEVICE_INFO_DESC(max_work_item_sizes<2>, range<2>);
template <> _LIBSYCL_DEVICE_INFO_DESC(max_work_item_sizes<3>, range<3>);
_LIBSYCL_DEVICE_INFO_DESC(max_work_group_size, std::size_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_char, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_short, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_int, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_long, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_long_long, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_float, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_double, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(preferred_vector_width_half, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_char, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_short, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_int, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_long, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_long_long, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_float, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_double, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(native_vector_width_half, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(max_clock_frequency, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(address_bits, std::uint32_t);
_LIBSYCL_DEVICE_INFO_DESC(max_mem_alloc_size, std::uint64_t);
_LIBSYCL_DEVICE_INFO_DESC(half_fp_config, std::vector<info::fp_config>);
_LIBSYCL_DEVICE_INFO_DESC(single_fp_config, std::vector<info::fp_config>);
_LIBSYCL_DEVICE_INFO_DESC(double_fp_config, std::vector<info::fp_config>);
_LIBSYCL_DEVICE_INFO_DESC(global_mem_size, std::uint64_t);
_LIBSYCL_DEVICE_INFO_DESC(local_mem_size, std::uint64_t);
_LIBSYCL_DEVICE_INFO_DESC(platform, sycl::platform);
_LIBSYCL_DEVICE_INFO_DESC(name, std::string);
_LIBSYCL_DEVICE_INFO_DESC(vendor, std::string);
_LIBSYCL_DEVICE_INFO_DESC(driver_version, std::string);

#undef _LIBSYCL_DEVICE_INFO_DESC
} // namespace device
} // namespace info

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_DEVICE_HPP
