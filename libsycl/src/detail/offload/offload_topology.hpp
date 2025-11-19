//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_OFFLOAD_TOPOLOGY
#define _LIBSYCL_OFFLOAD_TOPOLOGY

#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <cassert>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// Minimal span-like view.
template <class T> struct range_view {
  T *ptr{};
  size_t len{};
  T *begin() const { return ptr; }
  T *end() const { return ptr + len; }
  T &operator[](size_t i) const { return ptr[i]; }
  size_t size() const { return len; }
};

using Platform2DevContainer =
    std::vector<std::pair<ol_platform_handle_t, ol_device_handle_t>>;

/// Contiguous global storage of platform handlers and device handles (grouped
/// by platform) for a backend.
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

  /// Returns all platforms associated with this topology.
  ///
  /// \returns minimal span-like view to platforms associated with this
  /// topology.
  range_view<const ol_platform_handle_t> getPlatforms() const;

  /// Returns all devices associated with specific platform.
  ///
  /// \param PlatformId is index into MPlatforms.
  ///
  /// \returns minimal span-like view to devices associated with specified
  /// platform.
  range_view<ol_device_handle_t> getDevices(size_t PlatformId) const;

  /// Register new platform and devices into this topology.
  ///
  /// \param PlatformsAndDev collection of platforms & devices.
  void registerNewPlatformsAndDevices(Platform2DevContainer &PlatformsAndDev);

private:
  ol_platform_backend_t MBackend = OL_PLATFORM_BACKEND_UNKNOWN;

  // Platforms and devices belonging to this backend (flattened)
  std::vector<ol_platform_handle_t> MPlatforms;

  // Devices are sorted by platform (guarantee from liboffload)
  std::vector<ol_device_handle_t> MDevices;

  // Vector holding range of devices for each platform (index is platform index
  // within Platforms), so MDeviceRange.size() == MPlatforms.size()
  std::vector<range_view<ol_device_handle_t>> MDeviceRange;
};

// Initialize the topologies by calling olIterateDevices.
void discoverOffloadDevices();

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_TOPOLOGY
