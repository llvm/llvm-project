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

platform_impl *
platform_impl::getOrMakePlatformImpl(ol_platform_handle_t Platform,
                                     size_t PlatformIndex) {
  const std::lock_guard<std::mutex> Guard(getPlatformMapMutex());

  std::vector<std::unique_ptr<platform_impl>> &PlatformCache =
      getPlatformCache();

  // If we've already seen this platform, return the impl
  for (const auto &PlatImpl : PlatformCache) {
    if (PlatImpl->getHandleRef() == Platform)
      return PlatImpl.get();
  }

  // Otherwise make the impl.
  std::unique_ptr<platform_impl> Result;
  Result = std::make_unique<platform_impl>(Platform, PlatformIndex);
  PlatformCache.emplace_back(std::move(Result));

  return PlatformCache.back().get();
}

std::vector<platform> platform_impl::getPlatforms() {
  discoverOffloadDevices();
  std::vector<platform> Platforms;
  for (const auto &Topo : getOffloadTopologies()) {
    size_t PlatformIndex = 0;
    for (const auto &OffloadPlatform : Topo.platforms()) {
      platform Platform = detail::createSyclObjFromImpl<platform>(
          *getOrMakePlatformImpl(OffloadPlatform, PlatformIndex++));
      Platforms.push_back(std::move(Platform));
    }
  }
  return Platforms;
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
