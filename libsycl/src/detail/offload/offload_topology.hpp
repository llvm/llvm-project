//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the OffloadTopology class, which is
/// used to iterate over liboffload platforms and devices.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_OFFLOAD_TOPOLOGY
#define _LIBSYCL_OFFLOAD_TOPOLOGY

#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <cstdint>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

struct OffloadDeviceDesc {
  ol_platform_handle_t Platform;
  ol_device_handle_t Device;
  uint32_t DriverId;
};

using Platform2DevContainer = std::vector<OffloadDeviceDesc>;

struct OffloadPlatformGroup {
  ol_platform_handle_t Platform;
  uint32_t DriverId;
  std::vector<ol_device_handle_t> Devices;
};

/// Storage of platform driver groups and their device handles for a backend.
struct OffloadTopology {
  OffloadTopology() : MBackend(OL_PLATFORM_BACKEND_UNKNOWN) {}
  OffloadTopology(ol_platform_backend_t OlBackend) : MBackend(OlBackend) {}

  /// Updates backend for this topology.
  ///
  /// \param B new backend value.
  void setBackend(ol_platform_backend_t B) { MBackend = B; }

  /// Queries backend of this topology.
  ///
  /// \returns backend of this topology.
  ol_platform_backend_t getBackend() const { return MBackend; }

  /// Returns all platform driver groups associated with this topology.
  ///
  /// \returns platform driver groups associated with this topology.
  const std::vector<OffloadPlatformGroup> &getPlatformGroups() const {
    return MPlatformGroups;
  }

  /// Registers platform driver groups and devices into this topology.
  ///
  /// \param PlatformsAndDev collection of platforms & devices.
  void
  registerNewPlatformsAndDevices(const Platform2DevContainer &PlatformsAndDev);

private:
  ol_platform_backend_t MBackend = OL_PLATFORM_BACKEND_UNKNOWN;

  std::vector<OffloadPlatformGroup> MPlatformGroups;
};

// Initialize the topologies by calling olIterateDevices.
void discoverOffloadDevices();

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_TOPOLOGY
