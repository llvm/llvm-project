//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <__atomic/atomic.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__mutex/once_flag.h>
#include <__pstl/backends/statpart.h>
#include <__utility/no_destroy.h>

#include "std_thread_sched.h"

#if defined(__APPLE__)
#  include <dispatch/dispatch.h>
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

// Type of the user function to apply to an iteration index.
using callback_t = void (*)(void* context, size_t chunk);

#if defined(__APPLE__)
// On Darwin platforms, we use libdispatch to provide parallel execution.

void __apply(size_t chunk_count, void* context, callback_t func) noexcept {
  _LIBCPP_ASSERT_INTERNAL(func != nullptr, "Attempting to apply a function which is a nullptr");
  ::dispatch_apply_f(chunk_count, DISPATCH_APPLY_AUTO, context, func);
}

#elif defined(_WIN32)
// On Windows platforms, we use the thread pool API to provide parallel execution.

// Context for the callbacks executed by the thread pool.
struct apply_win32_context {
  std::atomic<size_t> next_iteration;
  size_t total_iterations;
  void* context;
  callback_t func;
};

// Callback executed by the Windows thread pool for each work item.
// Since the thread pool API doesn't allow passing the index directly when submitting work,
// the callback peels off the next iteration index from the atomic counter in the context.
static void WINAPI apply_win32_callback(PTP_CALLBACK_INSTANCE, void* callback_context, PTP_WORK) noexcept {
  apply_win32_context& ctx = *static_cast<apply_win32_context*>(callback_context);
  size_t iteration         = ctx.next_iteration.fetch_add(1, std::memory_order_relaxed);
  if (iteration < ctx.total_iterations)
    ctx.func(ctx.context, iteration);
}

void __apply(size_t chunk_count, void* context, callback_t func) noexcept {
  _LIBCPP_ASSERT_INTERNAL(func != nullptr, "Attempting to apply a function which is a nullptr");
  if (chunk_count < 2) {
    // Not enough iterations to call into the thread pool APIs, execute serially.
    apply_serial(chunk_count, context, func);
    return;
  }

  // Initialize the shared work context and create the thread pool work object.
  apply_win32_context work_context{0, chunk_count, context, func};
  PTP_WORK work = CreateThreadpoolWork(apply_win32_callback, &work_context, nullptr);
  if (!work) {
    // In case of failure, fall back to serial execution.
    apply_serial(chunk_count, context, func);
    return;
  }

  // Submit the work `chunk_count-1` times to the thread pool.
  // The current calling thread will also participate, hence 1 less element is submitted.
  for (size_t i = 1; i < chunk_count; ++i)
    SubmitThreadpoolWork(work);

  // Participate in the work.
  while (true) {
    // Peel off the next iteration number to work on.
    size_t chunk = work_context.next_iteration.fetch_add(1, std::memory_order_relaxed);
    if (chunk >= chunk_count)
      break; // The iterations are already exhausted, no more work to do.
    func(context, chunk);
  }

  // Wait until all running callbacks have completed, cancelling any pending ones.
  WaitForThreadpoolWorkCallbacks(work, true);

  // Cleanup.
  CloseThreadpoolWork(work);
}

#elif _LIBCPP_HAS_THREADS
// Otherwise, if threading is enabled on this platform, use the task scheduler built on top of std::thread.

void __apply(size_t chunk_count, void* context, callback_t func) noexcept {
  _LIBCPP_ASSERT_INTERNAL(func != nullptr, "Attempting to apply a function which is a nullptr");
  // Scheduler is stored in __no_destroy<> instead of using _LIBCPP_NO_DESTROY to guarantee it's never destroyed
  // regardless of a compiler; not destroying is not an optimization, it's a matter of correctness here.
  static constinit std::once_flag once;
  static constinit std::__no_destroy<Scheduler> sched{std::__uninitialized_tag{}};
  std::call_once(once, []() { sched.__emplace(); });
  sched.__get().apply(chunk_count, context, func);
}

#else
// Finally, in case there's no threading available, apply the functor serially.
// Calls like these should not happen in practice, but we need to preserve the semantics regardless.

void __apply(size_t chunk_count, void* context, callback_t func) noexcept {
  _LIBCPP_ASSERT_INTERNAL(func != nullptr, "Attempting to apply a function which is a nullptr");
  for (size_t i = 0; i < chunk_count; ++i)
    func(context, i);
}

#endif

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
