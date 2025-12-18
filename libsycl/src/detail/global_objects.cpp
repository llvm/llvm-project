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

std::vector<detail::OffloadTopology> &getOffloadTopologies() {
  static std::vector<detail::OffloadTopology> Topologies(
      OL_PLATFORM_BACKEND_LAST);
  return Topologies;
}

std::vector<PlatformImplUPtr> &getPlatformCache() {
  static std::vector<PlatformImplUPtr> PlatformCache{};
  return PlatformCache;
}

void shutdown() {
  // No error reporting in shutdown
  std::ignore = olShutDown();
}

#ifdef _WIN32
extern "C" _LIBSYCL_EXPORT BOOL WINAPI DllMain(HINSTANCE hinstDLL,
                                               DWORD fdwReason,
                                               LPVOID lpReserved) {
  // Perform actions based on the reason for calling.
  switch (fdwReason) {
  case DLL_PROCESS_DETACH:
    try {
      shutdown();
    } catch (std::exception &e) {
      // report
    }

    break;
  case DLL_PROCESS_ATTACH:
    break;
  case DLL_THREAD_ATTACH:
    break;
  case DLL_THREAD_DETACH:
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}
#else
// Setting low priority on destructor ensures it runs after all other global
// destructors. Priorities 0-100 are reserved by the compiler. The priority
// value 110 allows SYCL users to run their destructors after libsycl
// deinitialization.
__attribute__((destructor(110))) static void syclUnload() { shutdown(); }
#endif
} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
