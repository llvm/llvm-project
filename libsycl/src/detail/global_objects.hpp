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
class platform_impl;

// Offload topologies (one per backend) discovered from liboffload.
std::vector<detail::OffloadTopology> &getOffloadTopologies();

std::mutex &getPlatformMapMutex();
std::vector<std::unique_ptr<platform_impl>> &getPlatformCache();

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_GLOBAL_OBJECTS
