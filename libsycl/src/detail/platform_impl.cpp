//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_base.hpp>

#include <detail/global_objects.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

platform_impl &platform_impl::getPlatformImpl(ol_platform_handle_t Platform) {
  auto &PlatformCache = getPlatformCache();
  for (auto &PlatImpl : PlatformCache) {
    if (PlatImpl.getHandleRef() == Platform)
      return PlatImpl;
  }

  throw sycl::exception(
      sycl::make_error_code(sycl::errc::runtime),
      "Platform for requested handle can't be created. This handle is not in "
      "the list of platforms discovered by liboffload");
}

range_view<platform_impl> platform_impl::getPlatforms() {
  [[maybe_unused]] static auto InitPlatformsOnce = []() {
    discoverOffloadDevices();
    auto &PlatformCache = getPlatformCache();
    for (const auto &Topo : getOffloadTopologies()) {
      size_t PlatformIndex = 0;
      for (const auto &OffloadPlatform : Topo.platforms()) {
        PlatformCache.emplace_back(
            platform_impl(OffloadPlatform, PlatformIndex++));
      }
    }
    return true;
  }();
  auto &PlatformCache = getPlatformCache();
  return {PlatformCache.data(), PlatformCache.size()};
}

platform_impl::platform_impl(ol_platform_handle_t Platform,
                             size_t PlatformIndex)
    : MOffloadPlatform(Platform), MOffloadPlatformIndex(PlatformIndex) {
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  call_and_throw(olGetPlatformInfo, MOffloadPlatform, OL_PLATFORM_INFO_BACKEND,
                 sizeof(Backend), &Backend);
  MBackend = convertBackend(Backend);
  MOffloadBackend = Backend;
}
} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
