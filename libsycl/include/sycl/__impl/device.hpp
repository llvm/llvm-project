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

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class platform;

namespace detail {
class DeviceImpl;
} // namespace detail

// SYCL 2020 4.6.4. Device class.
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
  device();

  /// Constructs a SYCL device instance using the device
  /// identified by the provided device selector.
  /// \param DeviceSelector is SYCL 2020 device selector, a simple callable that
  /// takes a device and returns an int.
  template <
      typename DeviceSelector,
      // `DeviceImpl` (used as a parameter in private ctor) is incomplete
      // so would result in a error trying to instantiate
      // `EnableIfDeviceSelectorIsInvocable` below. Filter it out
      // before trying to do that.
      typename =
          std::enable_if_t<!std::is_same_v<DeviceSelector, detail::DeviceImpl>>,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit device(const DeviceSelector &deviceSelector)
      : device(detail::SelectDevice(deviceSelector)) {}

  /// Returns the backend associated with this device.
  ///
  /// \return the backend associated with this device.
  backend get_backend() const noexcept;

  /// Check if device is a CPU device.
  ///
  /// \return true if SYCL device is a CPU device.
  bool is_cpu() const;

  /// Check if device is a GPU device.
  ///
  /// \return true if SYCL device is a GPU device.
  bool is_gpu() const;

  /// Check if device is an accelerator device.
  ///
  /// \return true if SYCL device is an accelerator device.
  bool is_accelerator() const;

  /// Get associated SYCL platform.
  ///
  /// \return The associated SYCL platform.
  platform get_platform() const;

  /// Queries this SYCL device for information requested by the template
  /// parameter param.
  ///
  /// \return device info of type described in 4.6.4.4.
  template <typename Param>
  detail::is_device_info_desc_t<Param> get_info() const;

  /// Queries this SYCL device for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename detail::is_backend_info_desc<Param>::return_type
  get_backend_info() const;

  /// Queries which optional features this device supports (if any).
  ///
  /// \return true if this device has the given aspect.
  bool has(aspect asp) const;

  /// Partition device into sub devices.
  ///
  /// Available only when prop is info::partition_property::partition_equally.
  /// If this SYCL device does not support
  /// info::partition_property::partition_equally a feature_not_supported
  /// exception will be thrown.
  ///
  /// \param ComputeUnits is a desired count of compute units in each sub
  /// device.
  /// \return sub devices partitioned from this SYCL device equally based on the
  /// ComputeUnits parameter.
  template <info::partition_property prop>
  std::vector<device> create_sub_devices(size_t ComputeUnits) const;

  /// Partition device into sub devices.
  ///
  /// Available only when prop is info::partition_property::partition_by_counts.
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_counts a feature_not_supported
  /// exception will be thrown.
  ///
  /// \param Counts is a std::vector of desired compute units in sub devices.
  /// \return sub devices partitioned from this SYCL device by count sizes based
  /// on the Counts parameter.
  template <info::partition_property prop>
  std::vector<device>
  create_sub_devices(const std::vector<size_t> &Counts) const;

  /// Partition device into sub devices.
  ///
  /// Available only when prop is
  /// info::partition_property::partition_by_affinity_domain. If this SYCL
  /// device does not support
  /// info::partition_property::partition_by_affinity_domain or the SYCL device
  /// does not support provided info::affinity_domain provided a
  /// feature_not_supported exception will be thrown.
  ///
  /// \param AffinityDomain is one of the values described in Table 4.20 of the
  /// SYCL 2020 specification.
  /// \return sub devices partitioned from this SYCL device by affinity domain
  /// based on the AffinityDomain parameter.
  template <info::partition_property prop>
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const;

  /// Query available SYCL devices.
  ///
  /// \param deviceType is one of the values described in A.3 of the SYCL 2020
  /// specification.
  /// \return all SYCL devices available in the system of the device type
  /// specified.
  static std::vector<device>
  get_devices(info::device_type deviceType = info::device_type::all);

private:
  device(detail::DeviceImpl &Impl) : impl(&Impl) {}
  detail::DeviceImpl *impl;

  friend sycl::detail::ImplUtils;
}; // class device

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::device> : public sycl::detail::HashBase<sycl::device> {};

#endif // _LIBSYCL___IMPL_DEVICE_HPP
