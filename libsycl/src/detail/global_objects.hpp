//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

/// Returns offload topologies (one per backend) discovered from liboffload.
///
/// This vector is populated only once at the first call of get_platforms().
///
/// \returns std::vector of all offload topologies.
std::vector<detail::OffloadTopology> &getOffloadTopologies();

/// Returns implementation class objects for all platforms discovered from
/// liboffload.
///
/// This vector is populated only once at the first call of get_platforms().
///
/// \returns std::vector of implementation objects for all platforms.
std::vector<std::unique_ptr<PlatformImpl>> &getPlatformCache();

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_GLOBAL_OBJECTS
