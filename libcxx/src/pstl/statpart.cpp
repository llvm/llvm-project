//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__pstl/backends/statpart.h>
#include <atomic>
#include <utility>

#if defined(__APPLE__)
#  include <dispatch/dispatch.h>
#endif

#if defined(_WIN32)
#  include <windows.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

static void __apply_serial(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
  for (size_t i = 0; i < chunk_count; ++i)
    func(context, i);
}

#if defined(_WIN32)
static constinit thread_local bool __is_called_from_thread_pool = false;
static void __apply_win32(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
  if (chunk_count < 2 || __is_called_from_thread_pool) {
    __apply_serial(chunk_count, context, func);
    return;
  }

  struct work_context_t {
    std::atomic<size_t> iteration;
    size_t iterations_;
    void* context_;
    void (*func_)(void* context, size_t chunk);
  } work_context{0, chunk_count, context, func};

  auto callback = +[](PTP_CALLBACK_INSTANCE, void* callback_context, PTP_WORK) {
    work_context_t& ctx = *static_cast<work_context_t*>(callback_context);
    size_t iteration    = ctx.iteration.fetch_add(1, std::memory_order_relaxed);
    if (iteration < ctx.iterations_) {
      __is_called_from_thread_pool = true;
      ctx.func_(ctx.context_, iteration);
      __is_called_from_thread_pool = false;
    }
  };

  TP_WORK* work = CreateThreadpoolWork(callback, &work_context, nullptr);
  if (!work) {
    __apply_serial(chunk_count, context, func);
    return;
  }

  for (size_t i = 1; i < chunk_count; ++i)
    SubmitThreadpoolWork(work);
  for (size_t chunk; (chunk = work_context.iteration.fetch_add(1, std::memory_order_relaxed)) < chunk_count;)
    work_context.func_(work_context.context_, chunk);
  WaitForThreadpoolWorkCallbacks(work, FALSE);
  CloseThreadpoolWork(work);
}
#endif

void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
#if defined(__APPLE__)
  ::dispatch_apply_f(chunk_count, DISPATCH_APPLY_AUTO, context, func);
#elif defined(_WIN32)
  __apply_win32(chunk_count, context, func);
#else
  // Currently there's no parallelism provider on non-Apple platforms.
  // A basic task scheduler based on a thread pool will be added later to perform the application in parallel.
  __apply_serial(chunk_count, context, func);
#endif
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
