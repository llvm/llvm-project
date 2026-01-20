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
#include <unordered_map>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

void discoverOffloadDevices() {
  callAndThrow(olInit);

  using PerBackendDataType =
      std::array<std::pair<PlatformWithDevStorageType, size_t /*DevCount*/>,
                 OL_PLATFORM_BACKEND_LAST>;

  PerBackendDataType Mapping;
  // olIterateDevices() calls the lambda for each device. Devices that fail
  // probes or that report unknown backends are silently ignored.
  // TODO for debug purposes env variable can be added to report error at the
  // first failure and interrupt iteration.
  callNoCheck(
      olIterateDevices,
      [](ol_device_handle_t Dev, void *User) -> bool {
        auto *Data = static_cast<PerBackendDataType *>(User);
        ol_platform_handle_t Plat = nullptr;
        ol_result_t Res = callNoCheck(
            olGetDeviceInfo, Dev, OL_DEVICE_INFO_PLATFORM, sizeof(Plat), &Plat);
        // If an error occurs, ignore the device and continue iteration.
        if (Res != OL_SUCCESS)
          return true;

        ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;
        Res = callNoCheck(olGetPlatformInfo, Plat, OL_PLATFORM_INFO_BACKEND,
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

        auto &[Map, DevCount] = (*Data)[static_cast<size_t>(OlBackend)];
        Map[Plat].push_back(Dev);
        DevCount++;
        return true;
      },
      &Mapping);
  // Now register all platforms and devices into the topologies
  auto &OffloadTopologies = getOffloadTopologies();
  for (size_t I = 0; I < OL_PLATFORM_BACKEND_LAST; ++I) {
    OffloadTopology &Topo = OffloadTopologies[I];
    Topo.setBackend(static_cast<ol_platform_backend_t>(I));
    Topo.registerNewPlatformsAndDevices(Mapping[I].first, Mapping[I].second);
  }
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL
