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
#include <unordered_map>
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

using PlatformWithDevStorageType =
    std::unordered_map<ol_platform_handle_t, std::vector<ol_device_handle_t>>;

/// Contiguous global storage of platform handlers and device handles (grouped
/// by platform) for a backend.
struct OffloadTopology {
  OffloadTopology() : MBackend(OL_PLATFORM_BACKEND_UNKNOWN) {}
  OffloadTopology(ol_platform_backend_t OlBackend) : MBackend(OlBackend) {}

  /// Updates backend for this topology.
  ///
  /// \param B new backend value.
  void setBackend(ol_platform_backend_t B) { MBackend = B; }

  /// Returns all platforms associated with this topology.
  ///
  /// \returns minimal span-like view to platforms associated with this
  /// topology.
  range_view<const ol_platform_handle_t> platforms() const {
    return {MPlatforms.data(), MPlatforms.size()};
  }

  /// Returns all devices associated with specific platform.
  ///
  /// \param PlatformId platform_id is index into MPlatforms.
  ///
  /// \returns minimal span-like view to devices associated with specified
  /// platform.
  range_view<const ol_device_handle_t>
  devicesForPlatform(size_t PlatformId) const {
    if (PlatformId >= MDevRangePerPlatformId.size()) {
      assert(false && "Platform index exceeds number of platforms.");
      return {nullptr, 0};
    }
    return MDevRangePerPlatformId[PlatformId];
  }

  /// Register new platform and devices into this topology.
  ///
  /// \param PlatformsAndDev associative container with platforms & devices.
  /// \param TotalDevCount total device count for the platform.
  void
  registerNewPlatformsAndDevices(PlatformWithDevStorageType &PlatformsAndDev,
                                 size_t TotalDevCount) {
    if (!PlatformsAndDev.size())
      return;

    MPlatforms.reserve(PlatformsAndDev.size());
    MDevRangePerPlatformId.reserve(MPlatforms.size());
    MDevices.reserve(TotalDevCount);

    for (auto &[NewPlatform, NewDevs] : PlatformsAndDev) {
      MPlatforms.push_back(NewPlatform);
      range_view<const ol_device_handle_t> R{MDevices.data() + MDevices.size(),
                                             NewDevs.size()};
      MDevices.insert(MDevices.end(), NewDevs.begin(), NewDevs.end());
      MDevRangePerPlatformId.push_back(R);
    }

    assert(TotalDevCount == MDevices.size());
  }

  /// Queries backend of this topology.
  ///
  /// \returns backend of this topology.
  ol_platform_backend_t backend() const { return MBackend; }

private:
  ol_platform_backend_t MBackend = OL_PLATFORM_BACKEND_UNKNOWN;

  // Platforms and devices belonging to this backend (flattened)
  std::vector<ol_platform_handle_t> MPlatforms;
  std::vector<ol_device_handle_t> MDevices; // sorted by platform

  // Vector holding range of devices for each platform (index is platform index
  // within MPlatforms)
  std::vector<range_view<const ol_device_handle_t>>
      MDevRangePerPlatformId; // MDevRangePerPlatformId.size() ==
                              // MPlatforms.size()
};

// Initialize the topologies by calling olIterateDevices.
void discoverOffloadDevices();

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_TOPOLOGY
