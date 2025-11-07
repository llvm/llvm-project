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
  [[maybe_unused]] static auto DiscoverOnce = [&]() {
    call_and_throw(olInit);

    using PerBackendDataType =
        std::array<std::pair<PlatformWithDevStorageType, size_t /*DevCount*/>,
                   OL_PLATFORM_BACKEND_LAST>;

    PerBackendDataType Mapping;
    // olIterateDevices calls lambda for every device.
    // Returning early means jump to next iteration/next device.
    call_nocheck(
        olIterateDevices,
        [](ol_device_handle_t Dev, void *User) -> bool {
          auto *Data = static_cast<PerBackendDataType *>(User);
          ol_platform_handle_t Plat = nullptr;
          ol_result_t Res =
              call_nocheck(olGetDeviceInfo, Dev, OL_DEVICE_INFO_PLATFORM,
                           sizeof(Plat), &Plat);
          // If error occures, ignore platform and continue iteration
          if (Res != OL_SUCCESS)
            return true;

          ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;
          Res = call_nocheck(olGetPlatformInfo, Plat, OL_PLATFORM_INFO_BACKEND,
                             sizeof(OlBackend), &OlBackend);
          // If error occures, ignore platform and continue iteration
          if (Res != OL_SUCCESS)
            return true;

          // Skip host & unknown backends
          if (OL_PLATFORM_BACKEND_HOST == OlBackend ||
              OL_PLATFORM_BACKEND_UNKNOWN == OlBackend)
            return true;

          // Ensure backend index fits into array size
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
      Topo.set_backend(static_cast<ol_platform_backend_t>(I));
      Topo.registerNewPlatformsAndDevices(Mapping[I].first, Mapping[I].second);
    }

    return true;
  }();
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL
