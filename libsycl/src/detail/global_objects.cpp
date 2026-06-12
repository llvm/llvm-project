//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_objects.hpp>
#include <detail/offload/offload_utils.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager.hpp>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

GlobalHandler::StaticVarShutdownHandler::~StaticVarShutdownHandler() {
  ProgramAndKernelManager::getInstance().releaseResources();
  GlobalHandler::resetGlobalObjects();
}

void GlobalHandler::resetGlobalObjects() {
  getPlatformCache().clear();
  getOffloadTopologies() = {};

  // No error reporting in shutdown
  std::ignore = olShutDown();
}

void GlobalHandler::initPlatforms() {
  discoverOffloadDevices();

  registerStaticVarShutdownHandler();

  auto &PlatformCache = getPlatformCache();
  for (const auto &Topo : getOffloadTopologies()) {
    size_t PlatformIndex = 0;
    for (const auto &OffloadPlatform : Topo.getPlatforms()) {
      PlatformCache.emplace_back(std::make_unique<PlatformImpl>(
          OffloadPlatform, PlatformIndex++, PlatformImpl::PrivateTag{}));
    }
  }
}

std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST> &
GlobalHandler::getOffloadTopologies() {
  static std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST>
      Topologies{};
  return Topologies;
}

std::vector<PlatformImplUPtr> &GlobalHandler::getPlatformCache() {
  static std::vector<PlatformImplUPtr> PlatformCache{};
  return PlatformCache;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
