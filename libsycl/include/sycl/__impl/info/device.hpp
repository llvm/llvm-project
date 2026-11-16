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

struct device_type : detail::info_desc_tag<device_type, sycl::device> {
  using return_type = sycl::info::device_type;
};
struct vendor_id : detail::info_desc_tag<vendor_id, sycl::device> {
  using return_type = std::uint32_t;
};
struct max_compute_units
    : detail::info_desc_tag<max_compute_units, sycl::device> {
  using return_type = std::uint32_t;
};
template <int Dimensions = 3> struct max_work_item_sizes;
template <>
struct max_work_item_sizes<1>
    : detail::info_desc_tag<max_work_item_sizes<1>, sycl::device> {
  using return_type = range<1>;
};
template <>
struct max_work_item_sizes<2>
    : detail::info_desc_tag<max_work_item_sizes<2>, sycl::device> {
  using return_type = range<2>;
};
template <>
struct max_work_item_sizes<3>
    : detail::info_desc_tag<max_work_item_sizes<3>, sycl::device> {
  using return_type = range<3>;
};
struct max_work_group_size
    : detail::info_desc_tag<max_work_group_size, sycl::device> {
  using return_type = std::size_t;
};
struct preferred_vector_width_char
    : detail::info_desc_tag<preferred_vector_width_char, sycl::device> {
  using return_type = std::uint32_t;
};
struct preferred_vector_width_short
    : detail::info_desc_tag<preferred_vector_width_short, sycl::device> {
  using return_type = std::uint32_t;
};
struct preferred_vector_width_int
    : detail::info_desc_tag<preferred_vector_width_int, sycl::device> {
  using return_type = std::uint32_t;
};
struct preferred_vector_width_long
    : detail::info_desc_tag<preferred_vector_width_long, sycl::device> {
  using return_type = std::uint32_t;
};
struct preferred_vector_width_float
    : detail::info_desc_tag<preferred_vector_width_float, sycl::device> {
  using return_type = std::uint32_t;
};
struct preferred_vector_width_double
    : detail::info_desc_tag<preferred_vector_width_double, sycl::device> {
  using return_type = std::uint32_t;
};
struct preferred_vector_width_half
    : detail::info_desc_tag<preferred_vector_width_half, sycl::device> {
  using return_type = std::uint32_t;
};
struct native_vector_width_char
    : detail::info_desc_tag<native_vector_width_char, sycl::device> {
  using return_type = std::uint32_t;
};
struct native_vector_width_short
    : detail::info_desc_tag<native_vector_width_short, sycl::device> {
  using return_type = std::uint32_t;
};
struct native_vector_width_int
    : detail::info_desc_tag<native_vector_width_int, sycl::device> {
  using return_type = std::uint32_t;
};
struct native_vector_width_long
    : detail::info_desc_tag<native_vector_width_long, sycl::device> {
  using return_type = std::uint32_t;
};
struct native_vector_width_float
    : detail::info_desc_tag<native_vector_width_float, sycl::device> {
  using return_type = std::uint32_t;
};
struct native_vector_width_double
    : detail::info_desc_tag<native_vector_width_double, sycl::device> {
  using return_type = std::uint32_t;
};
struct native_vector_width_half
    : detail::info_desc_tag<native_vector_width_half, sycl::device> {
  using return_type = std::uint32_t;
};
struct max_clock_frequency
    : detail::info_desc_tag<max_clock_frequency, sycl::device> {
  using return_type = std::uint32_t;
};
struct address_bits : detail::info_desc_tag<address_bits, sycl::device> {
  using return_type = std::uint32_t;
};
struct max_mem_alloc_size
    : detail::info_desc_tag<max_mem_alloc_size, sycl::device> {
  using return_type = std::uint64_t;
};
struct half_fp_config : detail::info_desc_tag<half_fp_config, sycl::device> {
  using return_type = std::vector<info::fp_config>;
};
struct single_fp_config
    : detail::info_desc_tag<single_fp_config, sycl::device> {
  using return_type = std::vector<info::fp_config>;
};
struct double_fp_config
    : detail::info_desc_tag<double_fp_config, sycl::device> {
  using return_type = std::vector<info::fp_config>;
};
struct global_mem_size : detail::info_desc_tag<global_mem_size, sycl::device> {
  using return_type = std::uint64_t;
};
struct local_mem_size : detail::info_desc_tag<local_mem_size, sycl::device> {
  using return_type = std::uint64_t;
};
struct platform : detail::info_desc_tag<platform, sycl::device> {
  using return_type = sycl::platform;
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

} // namespace device
} // namespace info

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_DEVICE_HPP
