//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the DeviceImpl class, which implements
/// sycl::device functionality.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_SRC_DETAIL_DEVICE_IMPL_HPP
#define _LIBSYCL_SRC_DETAIL_DEVICE_IMPL_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/device.hpp>

#include <detail/offload/offload_utils.hpp>
#include <detail/platform_impl.hpp>

#include <OffloadAPI.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

template <typename T> struct is_range : std::false_type {};
template <int Dimensions>
struct is_range<range<Dimensions>> : std::true_type {};
template <typename T> inline constexpr bool is_range_v = is_range<T>::value;

class DeviceImpl {
  // Helper to limit DeviceImpl creation. It must be created in platform ctor
  // only. Using tag instead of private ctor + friend class to allow make_unique
  // usage and to align with classes which impl is shared_ptr<>.
  struct PrivateTag {
    explicit PrivateTag() = default;
  };
  friend class PlatformImpl;

public:
  /// Constructs a SYCL device instance using the provided
  /// offload device instance.
  ///
  /// \param Device is a raw offload library handle representing device.
  /// \param Platform is a platform this device belongs to.
  /// All device impls must be created in corresponding platform ctor.
  explicit DeviceImpl(ol_device_handle_t Device, PlatformImpl &Platform,
                      PrivateTag)
      : MOffloadDevice(Device), MPlatform(Platform) {}

  ~DeviceImpl() = default;

  /// Queries device type from offloading runtime
  ///
  /// \return device type of the device
  info::device_type getDeviceType() const;

  /// Check if device is a CPU device
  ///
  /// \return true if SYCL device is a CPU device
  bool isCPU() const;

  /// Check if device is a GPU device
  ///
  /// \return true if SYCL device is a GPU device
  bool isGPU() const;

  /// Check if device is an accelerator device
  ///
  /// \return true if SYCL device is an accelerator device
  bool isAccelerator() const;

  /// Returns the backend associated with this device.
  ///
  /// \return the sycl::backend associated with this device.
  backend getBackend() const noexcept;

  /// Returns the implementation class object of platform associated with this
  /// device.
  ///
  /// \return platform implementation object this device belongs to.
  PlatformImpl &getPlatformImpl() const { return MPlatform; }

  /// Checks if this device supports aspect.
  ///
  /// \param Aspect to perform a check of.
  /// \return true if this device has the given aspect.
  bool has(aspect Aspect) const;

  /// Queries this device for information requested by the template parameter
  /// param.
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type getInfo() const {
    // TODO implement caching
    using namespace info::device;
    using Map = InfoOLMapping<ol_device_info_t>;

    constexpr ol_device_info_t OLInfo = mapInfoDesc<Param, ol_device_info_t>(
        Map::M<device_type>{OL_DEVICE_INFO_TYPE},
        Map::M<vendor_id>{OL_DEVICE_INFO_VENDOR_ID},
        Map::M<max_compute_units>{OL_DEVICE_INFO_NUM_COMPUTE_UNITS},
        Map::M<max_work_item_sizes<1>>{
            OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION},
        Map::M<max_work_item_sizes<2>>{
            OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION},
        Map::M<max_work_item_sizes<3>>{
            OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION},
        Map::M<max_work_group_size>{OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE},
        Map::M<preferred_vector_width_char>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR},
        Map::M<preferred_vector_width_short>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT},
        Map::M<preferred_vector_width_int>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT},
        Map::M<preferred_vector_width_long>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG},
        Map::M<preferred_vector_width_float>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT},
        Map::M<preferred_vector_width_double>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE},
        Map::M<preferred_vector_width_half>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF},
        Map::M<native_vector_width_char>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR},
        Map::M<native_vector_width_short>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT},
        Map::M<native_vector_width_int>{OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT},
        Map::M<native_vector_width_long>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG},
        Map::M<native_vector_width_float>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT},
        Map::M<native_vector_width_double>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE},
        Map::M<native_vector_width_half>{
            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF},
        Map::M<max_clock_frequency>{OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY},
        Map::M<address_bits>{OL_DEVICE_INFO_ADDRESS_BITS},
        Map::M<max_mem_alloc_size>{OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE},
        Map::M<half_fp_config>{OL_DEVICE_INFO_HALF_FP_CONFIG},
        Map::M<single_fp_config>{OL_DEVICE_INFO_SINGLE_FP_CONFIG},
        Map::M<double_fp_config>{OL_DEVICE_INFO_DOUBLE_FP_CONFIG},
        Map::M<global_mem_size>{OL_DEVICE_INFO_GLOBAL_MEM_SIZE},
        Map::M<local_mem_size>{OL_DEVICE_INFO_WORK_GROUP_LOCAL_MEM_SIZE},
        Map::M<name>{OL_DEVICE_INFO_PRODUCT_NAME},
        Map::M<vendor>{OL_DEVICE_INFO_VENDOR},
        Map::M<driver_version>{OL_DEVICE_INFO_DRIVER_VERSION});

    using RetT = typename Param::return_type;

    // max_work_group_size returns std::size_t while its liboffload
    // counterpart returns uint32_t.
    if constexpr (std::is_same_v<Param, max_work_group_size>) {
      return static_cast<RetT>(getOLInfoAs<uint32_t>(OLInfo));
    } else if constexpr (std::is_arithmetic_v<RetT>) {
      return getOLInfoAs<RetT>(OLInfo);
    } else if constexpr (std::is_same_v<RetT, std::vector<info::fp_config>>) {
      return convertFPConfigToSYCL(
          getOLInfoAs<ol_device_fp_capability_flags_t>(OLInfo));
    } else if constexpr (std::is_same_v<RetT, info::device_type>) {
      return convertDeviceTypeToSYCL(getOLInfoAs<ol_device_type_t>(OLInfo));
    } else if constexpr (std::is_same_v<RetT, std::string>) {
      size_t ExpectedSize = 0;
      callAndThrow(olGetDeviceInfoSize, MOffloadDevice, OLInfo, &ExpectedSize);
      assert(ExpectedSize > 0 && "String info descriptor size must account for "
                                 "the null terminator");
      std::string Result;
      // liboffload counts null terminator in the size while std::string
      // doesn't.
      Result.resize(ExpectedSize - 1);
      callAndThrow(olGetDeviceInfo, MOffloadDevice, OLInfo, ExpectedSize,
                   Result.data());
      return Result;
    } else if constexpr (is_range_v<RetT>) {
      auto Dims = getOLInfoAs<ol_dimensions_t>(OLInfo);
      // The last SYCL dimension is the fastest varying one, which corresponds
      // to the first dimension reported by liboffload.
      if constexpr (RetT::dimensions == 1)
        return RetT{Dims.x};
      else if constexpr (RetT::dimensions == 2)
        return RetT{Dims.y, Dims.x};
      else
        return RetT{Dims.z, Dims.y, Dims.x};
    } else {
      static_assert(AlwaysFalse<Param>,
                    "Info descriptor is not properly supported");
    }
  }

  /// \return the corresponding liboffload device handle.
  ol_device_handle_t getOLHandle() const { return MOffloadDevice; }

private:
  /// Queries liboffload for the Info property and returns it as a T value.
  template <typename T> T getOLInfoAs(ol_device_info_t Info) const {
    T Value{};
    callAndThrow(olGetDeviceInfo, MOffloadDevice, Info, sizeof(Value), &Value);
    return Value;
  }

  ol_device_handle_t MOffloadDevice = {};
  PlatformImpl &MPlatform;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_SRC_DETAIL_DEVICE_IMPL_HPP
