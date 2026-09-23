//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/global_objects.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager.hpp>
#include <detail/queue_impl.hpp>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <cassert>
#include <tuple>
#include <utility>
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
    ProgramAndKernelManager::getInstance().releaseResources();
    // No error reporting in shutdown
    std::ignore = olShutDown();
  }
};

void registerStaticVarShutdownHandler() {
  // Touch the program manager singleton first: static objects are destroyed in
  // reverse order of construction, so this guarantees it is still alive when
  // ~StaticVarShutdownHandler() calls releaseResources() on it.
  std::ignore = ProgramAndKernelManager::getInstance();
  static StaticVarShutdownHandler handler{};
}

std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST> &
getOffloadTopologies() {
  static std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST>
      Topologies{};
  return Topologies;
}

std::vector<PlatformImplUPtr> &getPlatformCache() {
  static std::vector<PlatformImplUPtr> PlatformCache{};
  return PlatformCache;
}

InstanceWithLock<AsyncExceptionsContainer> &getAsyncExceptionList() {
  static InstanceWithLock<AsyncExceptionsContainer> AsyncExceptionList;
  return AsyncExceptionList;
}

void recordAsyncException(const std::shared_ptr<QueueImpl> &QueuePtr,
                          const std::exception_ptr &ExceptionPtr) {
  assert(QueuePtr && "Queue impl ptr can't be nullptr");
  AsyncExceptionKey Key{QueuePtr, QueuePtr->getContextWeakPtr()};

  auto &[AsyncExceptions, AsyncExceptionsMutex] = getAsyncExceptionList();
  std::lock_guard<SpinLock> Lock(AsyncExceptionsMutex);
  addAsyncException(AsyncExceptions[std::move(Key)], ExceptionPtr);
}

void flushAsyncExceptions() {
  auto &[AsyncExceptions, AsyncExceptionsMutex] = getAsyncExceptionList();
  AsyncExceptionsContainer AsyncExceptionsSwap;
  {
    std::lock_guard<SpinLock> Lock(AsyncExceptionsMutex);
    std::swap(AsyncExceptions, AsyncExceptionsSwap);
  }

  for (auto &[EntryKey, ExceptionList] : AsyncExceptionsSwap) {
    exception_list Exceptions = std::move(ExceptionList);

    if (Exceptions.size() == 0)
      continue;

    // SYCL 2020 4.13.1.3. Priorities of async handlers: the handler the queue
    // was constructed with comes first, the handler of the context enclosing
    // the queue comes next.
    const auto &[WeakQueue, WeakContext] = EntryKey;

    if (std::shared_ptr<QueueImpl> Queue = WeakQueue.lock();
        Queue && Queue->getAsyncHandler()) {
      Queue->getAsyncHandler()(std::move(Exceptions));
      continue;
    }

    if (std::shared_ptr<ContextImpl> Context = WeakContext.lock();
        Context && Context->get_async_handler()) {
      Context->get_async_handler()(std::move(Exceptions));
      continue;
    }

    // Neither the queue nor the context has a handler, or both of them are
    // dead. A context constructed without an async_handler is given the default
    // one at construction, so there is no need for a context to carry an empty
    // handler: leaving it empty would end up here with an identical result.
    defaultAsyncHandler(std::move(Exceptions));
  }
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
