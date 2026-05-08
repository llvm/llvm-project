//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_objects.hpp>
#include <detail/platform_impl.hpp>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {
// libsycl follows SYCL 2020 specification that doesn't declare any
// init/shutdown methods that can help to avoid usage of static variables.
// liboffload uses static variables too. In the first call of get_platforms
// we call liboffload's iterateDevices that leads to liboffload static
// storage initialization. Then we initialize our own local static var of
// StaticVarShutdownHandler type to be able to call our shutdown methods
// earlier and before the liboffload objects are destructed at the end of
// program. See documentation of std::exit for local objects with static
// storage duration.
struct StaticVarShutdownHandler {
  StaticVarShutdownHandler(const StaticVarShutdownHandler &) = delete;
  StaticVarShutdownHandler &
  operator=(const StaticVarShutdownHandler &) = delete;
  ~StaticVarShutdownHandler() {
    // No error reporting in shutdown
    std::ignore = olShutDown();
  }
};

void registerStaticVarShutdownHandler() {
  static StaticVarShutdownHandler handler{};
}

std::vector<detail::OffloadTopology> &getOffloadTopologies() {
  static std::vector<detail::OffloadTopology> Topologies(
      OL_PLATFORM_BACKEND_LAST);
  return Topologies;
}

std::vector<PlatformImplUPtr> &getPlatformCache() {
  static std::vector<PlatformImplUPtr> PlatformCache{};
  return PlatformCache;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
