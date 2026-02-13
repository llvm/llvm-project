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
