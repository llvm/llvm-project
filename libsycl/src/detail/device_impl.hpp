//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_IMPL
#define _LIBSYCL_DEVICE_IMPL

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/device.hpp>

#include <detail/offload/offload_utils.hpp>
#include <detail/platform_impl.hpp>

#include <OffloadAPI.h>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

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
    using namespace info::device;
    using Map = info_ol_mapping<ol_device_info_t>;

    constexpr ol_device_info_t olInfo = map_info_desc<Param, ol_device_info_t>(
        Map::M<device_type>{OL_DEVICE_INFO_TYPE},
        Map::M<name>{OL_DEVICE_INFO_NAME},
        Map::M<vendor>{OL_DEVICE_INFO_VENDOR},
        Map::M<driver_version>{OL_DEVICE_INFO_DRIVER_VERSION});

    size_t ExpectedSize = 0;
    callAndThrow(olGetDeviceInfoSize, MOffloadDevice, olInfo, &ExpectedSize);

    if constexpr (std::is_same_v<typename Param::return_type, std::string>) {
      std::string Result;
      // liboffload counts null terminator in the size while std::string
      // doesn't.
      Result.resize(ExpectedSize - 1);
      callAndThrow(olGetDeviceInfo, MOffloadDevice, olInfo, ExpectedSize,
                   Result.data());
      return Result;
    } else if constexpr (olInfo == OL_DEVICE_INFO_TYPE) {
      assert((sizeof(typename Param::return_type) == ExpectedSize) &&
             "Size of info descriptor reported by backend doesn't match with "
             "expected.");
      ol_device_type_t olType{};
      callAndThrow(olGetDeviceInfo, MOffloadDevice, olInfo, sizeof(olType),
                   &olType);
      return convertDeviceTypeToSYCL(olType);
    } else
      static_assert(false && "Info descriptor is not properly supported");
  }

  ol_device_handle_t getOLHandle() { return MOffloadDevice; }

private:
  ol_device_handle_t MOffloadDevice = {};
  PlatformImpl &MPlatform;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_IMPL
