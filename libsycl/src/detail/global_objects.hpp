//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of all the global objects of libsycl.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_GLOBAL_OBJECTS
#define _LIBSYCL_GLOBAL_OBJECTS

#include <detail/offload/offload_topology.hpp>
#include <sycl/__impl/detail/config.hpp>

#include <memory>
#include <mutex>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class PlatformImpl;

class GlobalHandler {
public:
  /// Returns offload topologies (one per backend) discovered from liboffload.
  ///
  /// This array is populated only once at the first call of get_platforms().
  ///
  /// \returns std::array of all offload topologies.
  static std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST> &
  getOffloadTopologies();

  /// Returns implementation class objects for all platforms discovered from
  /// liboffload.
  ///
  /// This vector is populated only once at the first call of get_platforms().
  ///
  /// \returns std::vector of implementation objects for all platforms.
  static std::vector<std::unique_ptr<PlatformImpl>> &getPlatformCache();

protected:
  GlobalHandler() = delete;

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
    ~StaticVarShutdownHandler();
  };

  static void registerStaticVarShutdownHandler() {
    static StaticVarShutdownHandler handler{};
  }

  // These methods and 2 friends declarations below are needed to be able to
  // call them in tests to reset global state of libsycl and liboffload between
  // tests. During normal execution initPlatforms is called once from
  // get_platforms and resetGlobalObjects is called once during static variables
  // destruction at the end of program.
  static void initPlatforms();
  static void resetGlobalObjects();

  friend class PlatformImpl;
  friend class UnittestsHelper;
};

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_GLOBAL_OBJECTS
