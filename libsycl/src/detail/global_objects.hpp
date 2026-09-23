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
#include <detail/spinlock.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/exception.hpp>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class ContextImpl;
class PlatformImpl;
class QueueImpl;

template <typename T> using InstanceWithLock = std::pair<T, SpinLock>;

/// Returns offload topologies (one per backend) discovered from liboffload.
///
/// This vector is populated only once at the first call of get_platforms().
///
/// \returns std::array of all offload topologies.
std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST> &
getOffloadTopologies();

/// Returns implementation class objects for all platforms discovered from
/// liboffload.
///
/// This vector is populated only once at the first call of get_platforms().
///
/// \returns std::vector of implementation objects for all platforms.
std::vector<std::unique_ptr<PlatformImpl>> &getPlatformCache();

// This initializes a function-local variable whose destructor is invoked as
// the SYCL shared library is first being unloaded.
void registerStaticVarShutdownHandler();

using AsyncExceptionKey =
    std::pair<std::weak_ptr<QueueImpl>, std::weak_ptr<ContextImpl>>;

struct AsyncExceptionKeyOwnerLess {
  bool operator()(const AsyncExceptionKey &LHS,
                  const AsyncExceptionKey &RHS) const noexcept {
    std::owner_less<std::weak_ptr<QueueImpl>> QueueLess;
    if (QueueLess(LHS.first, RHS.first))
      return true;
    if (QueueLess(RHS.first, LHS.first))
      return false;
    return std::owner_less<std::weak_ptr<ContextImpl>>{}(LHS.second,
                                                         RHS.second);
  }
};

using AsyncExceptionsContainer =
    std::map<AsyncExceptionKey, exception_list, AsyncExceptionKeyOwnerLess>;

/// \return pair of container with unreported exceptions and an unlocked
/// SpinLock.
InstanceWithLock<AsyncExceptionsContainer> &getAsyncExceptionList();

/// Adds an exception to the list of unreported asynchronous exceptions
/// associated with the given queue and with the context enclosing it.
void recordAsyncException(const std::shared_ptr<QueueImpl> &QueuePtr,
                          const std::exception_ptr &ExceptionPtr);

/// Reports all unreported asynchronous exceptions and clears the list.
void flushAsyncExceptions();

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_GLOBAL_OBJECTS
