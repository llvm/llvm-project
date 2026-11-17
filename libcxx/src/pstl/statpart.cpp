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

#if defined(__APPLE__)
#  include <dispatch/dispatch.h>
#endif

#if defined(_WIN32)
#  include <windows.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

// Type of the user function to apply to an iteration index.
using callback_t = void (*)(void* context, size_t chunk);

// A fallback implementation of __apply(): just apply the function serially.
[[maybe_unused]] static void apply_serial(size_t chunk_count, void* context, callback_t func) noexcept {
  for (size_t i = 0; i < chunk_count; ++i)
    func(context, i);
}

#if defined(__APPLE__)
// On Darwin platforms, we use libdispatch to provide parallel execution.

void __apply(size_t chunk_count, void* context, callback_t func) noexcept {
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

#else
// Currently there's no parallelism provider on other platforms.
// A basic task scheduler based on a thread pool will be added later to perform the application in parallel.

void __apply(size_t chunk_count, void* context, callback_t func) noexcept { apply_serial(chunk_count, context, func); }

#endif

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
