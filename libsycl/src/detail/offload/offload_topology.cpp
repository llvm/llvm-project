//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_objects.hpp>
#include <detail/offload/offload_topology.hpp>
#include <detail/offload/offload_utils.hpp>

#include <array>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// Platforms for this backend
range_view<const ol_platform_handle_t> OffloadTopology::getPlatforms() const {
  return {MPlatforms.data(), MPlatforms.size()};
}

// Devices for a specific platform (PlatformId is index into Platforms)
range_view<ol_device_handle_t>
OffloadTopology::getDevices(size_t PlatformId) const {
  if (PlatformId >= MDeviceRange.size()) {
    return {nullptr, 0};
  }
  return MDeviceRange[PlatformId];
}

void OffloadTopology::registerNewPlatformsAndDevices(
    Platform2DevContainer &PlatformsAndDev) {
  if (!PlatformsAndDev.size())
    return;

  // MDeviceRange is populated with iterators of MDevices. Allocate required
  // space in advance to keep them valid.
  MDevices.reserve(PlatformsAndDev.size());

  for (auto &[Platform, NewDev] : PlatformsAndDev) {
    MDevices.push_back(NewDev);

    // Platform is not unique within PlatformsAndDev but the container is sorted
    if (MPlatforms.empty() || MPlatforms.back() != Platform) {
      MPlatforms.push_back(Platform);
      range_view<ol_device_handle_t> R{&MDevices.back(), 1 /*Size == 1*/};
      MDeviceRange.push_back(R);
    } else {
      // Device is inserted already, just increment device count for the current
      // platform
      MDeviceRange.back().len++;
    }
  }
}

void discoverOffloadDevices() {
  callAndThrow(olInit, nullptr);

  // liboffload returns devices sorted by backend + platform. We rely on this
  // behavior during device enumeration.
  using PerBackendDataType =
      std::array<Platform2DevContainer, OL_PLATFORM_BACKEND_LAST>;

  PerBackendDataType Mapping;
  // olIterateDevices() calls the lambda for each device. Devices that fail
  // probes or that report unknown backends are silently ignored.
  // TODO for debug purposes env variable can be added to report error at the
  // first failure and interrupt iteration.
  callNoCheck(
      olIterateDevices,
      [](ol_device_handle_t Dev, void *UserData) -> bool {
        auto *Data = static_cast<PerBackendDataType *>(UserData);

        ol_platform_handle_t Platform = nullptr;
        ol_result_t Res =
            callNoCheck(olGetDeviceInfo, Dev, OL_DEVICE_INFO_PLATFORM,
                        sizeof(Platform), &Platform);
        // If an error occurs, ignore the device and continue iteration.
        if (Res != OL_SUCCESS)
          return true;

        ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;
        Res = callNoCheck(olGetPlatformInfo, Platform, OL_PLATFORM_INFO_BACKEND,
                          sizeof(OlBackend), &OlBackend);
        // If an error occurs, ignore the device and continue iteration.
        if (Res != OL_SUCCESS)
          return true;

        // Ignore host and unknown backends
        if (OL_PLATFORM_BACKEND_HOST == OlBackend ||
            OL_PLATFORM_BACKEND_UNKNOWN == OlBackend)
          return true;

        // Ignore the device if the backend index exceeds the number of backends
        // known at compile time. This should only happen when running with a
        // newer version of liboffload than libsycl was compiled with.
        if (OlBackend >= OL_PLATFORM_BACKEND_LAST)
          return true;

        (*Data)[static_cast<size_t>(OlBackend)].push_back({Platform, Dev});
        return true;
      },
      &Mapping);
  // Now register all platforms and devices into the topologies
  auto &OffloadTopologies = getOffloadTopologies();
  for (size_t I = 0; I < OL_PLATFORM_BACKEND_LAST; ++I) {
    OffloadTopology &Topo = OffloadTopologies[I];
    Topo.setBackend(static_cast<ol_platform_backend_t>(I));
    Topo.registerNewPlatformsAndDevices(Mapping[I]);
  }
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL
