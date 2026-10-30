//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 device class, which
/// represents a single SYCL device on which kernels can be executed.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DEVICE_HPP
#define _LIBSYCL___IMPL_DEVICE_HPP

#include <sycl/__impl/aspect.hpp>
#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/device_selector.hpp>
#include <sycl/__impl/info/device.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class platform;

namespace detail {
class DeviceImpl;
} // namespace detail

/// A device on which kernels can be executed.
///
/// \sa SYCL 2020 4.6.4. Device class.
class _LIBSYCL_EXPORT device {
public:
  device(const device &rhs) = default;

  device(device &&rhs) = default;

  device &operator=(const device &rhs) = default;

  device &operator=(device &&rhs) = default;

  friend bool operator==(const device &lhs, const device &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const device &lhs, const device &rhs) {
    return !(lhs == rhs);
  }

  /// Constructs a SYCL device instance using the default device (device chosen
  /// by default device selector).
  ///
  /// \throw sycl::exception with sycl::errc::runtime if no device is available.
  device() : device(default_selector_v) {}

  /// Constructs a SYCL device instance using the device
  /// identified by the provided device selector.
  /// \param deviceSelector a callable that takes a device and returns an int
  /// score; the device with the highest score is chosen, and a negative score
  /// rejects a device.
  /// \throw sycl::exception with sycl::errc::runtime if \p deviceSelector
  /// rejects all devices.
  template <
      typename DeviceSelector,
      // `DeviceImpl` (used as a parameter in private ctor) is incomplete
      // so would result in an error trying to instantiate
      // `EnableIfDeviceSelectorIsInvocable` below. Filter it out
      // before trying to do that.
      typename =
          std::enable_if_t<!std::is_same_v<DeviceSelector, detail::DeviceImpl>>,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit device(const DeviceSelector &deviceSelector)
      : device(detail::SelectDevice(deviceSelector)) {}

  /// \return the backend associated with this device.
  backend get_backend() const noexcept;

  /// \return true if this device is a CPU device.
  bool is_cpu() const;

  /// \return true if this device is a GPU device.
  bool is_gpu() const;

  /// \return true if this device is an accelerator device.
  bool is_accelerator() const;

  /// \return the associated SYCL platform.
  platform get_platform() const;

  /// \return the value of the device information descriptor Param (SYCL 2020
  /// 4.6.4.4. Information descriptors).
  template <typename Param>
  detail::is_device_info_desc_t<Param> get_info() const;

  /// \return the value of the SYCL backend-specific information descriptor
  /// Param of this device.
  template <typename Param>
  typename detail::is_backend_info_desc<Param>::return_type
  get_backend_info() const;

  /// Queries which optional features this device supports (if any).
  ///
  /// \param asp the aspect to check for (SYCL 2020 4.6.4.5. Aspects).
  /// \return true if this device has \p asp.
  bool has(aspect asp) const;

  /// Partition device into sub devices.
  ///
  /// Available only when Prop is info::partition_property::partition_equally.
  /// If this SYCL device does not support
  /// info::partition_property::partition_equally a feature_not_supported
  /// exception will be thrown.
  ///
  /// \param count is a desired count of compute units in each sub
  /// device.
  /// \return sub devices partitioned from this SYCL device equally based on the
  /// count parameter.
  template <info::partition_property Prop>
  std::vector<device> create_sub_devices(std::size_t count) const;

  /// Partition device into sub devices.
  ///
  /// Available only when Prop is info::partition_property::partition_by_counts.
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_counts a feature_not_supported
  /// exception will be thrown.
  ///
  /// \param counts is a std::vector of desired compute units in sub devices.
  /// \return sub devices partitioned from this SYCL device by count sizes based
  /// on the counts parameter.
  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(const std::vector<std::size_t> &counts) const;

  /// Partition device into sub devices.
  ///
  /// Available only when Prop is
  /// info::partition_property::partition_by_affinity_domain. If this SYCL
  /// device does not support
  /// info::partition_property::partition_by_affinity_domain or the SYCL device
  /// does not support the provided info::partition_affinity_domain, a
  /// feature_not_supported exception will be thrown.
  ///
  /// \param affinityDomain the affinity domain to partition this device along
  /// (SYCL 2020 4.6.4.7.3. Partition affinity domain).
  /// \return sub devices partitioned from this SYCL device by affinity domain
  /// based on the affinityDomain parameter.
  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const;

  /// Query available SYCL devices.
  ///
  /// \param type the type of the devices to return (SYCL 2020 4.6.4.7.1. Device
  /// type).
  /// \return all SYCL devices available in the system of the device type
  /// specified.
  static std::vector<device>
  get_devices(info::device_type type = info::device_type::all);

private:
  device(detail::DeviceImpl &Impl) : impl(&Impl) {}
  detail::DeviceImpl *impl;

  friend sycl::detail::ImplUtils;
}; // class device

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::device> : public sycl::detail::HashBase<sycl::device> {};

#endif // _LIBSYCL___IMPL_DEVICE_HPP
