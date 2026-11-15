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

// A fallback implementation of __apply(): just apply the function serially.
[[maybe_unused]] static void
__apply_serial(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
  for (size_t i = 0; i < chunk_count; ++i)
    func(context, i);
}

#if defined(_WIN32)
// This flag is set on before calling out the user function.
static constinit thread_local bool __apply_win32_reentrant = false;

// Context for the callbacks executed by the thread pool.
struct __apply_win32_context {
  std::atomic<size_t> iteration;
  size_t iterations;
  void* context;
  void (*func)(void* context, size_t chunk);
};

// Callback executed by the Windows thread pool for each work item.
// Since the thread pool API doesn't allow passing the index directly when submitting work,
// the callback peels off the next iteration index from the atomic counter in the context.
static void WINAPI __apply_win32_callback(PTP_CALLBACK_INSTANCE, void* callback_context, PTP_WORK) noexcept {
  __apply_win32_context& ctx = *static_cast<__apply_win32_context*>(callback_context);
  size_t iteration           = ctx.iteration.fetch_add(1, std::memory_order_relaxed);
  if (iteration < ctx.iterations) {
    __apply_win32_reentrant = true;
    ctx.func(ctx.context, iteration);
    __apply_win32_reentrant = false;
  }
}

static void __apply_win32(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
  if (chunk_count < 2 || __apply_win32_reentrant) {
    // It's possible to deadlock the default thread pool if we allow nested parallelism.
    // For now explicitly prohibit it and execute such requests serially.
    __apply_serial(chunk_count, context, func);
    return;
  }

  // Initialize the shared work context and create the thread pool work object.
  __apply_win32_context work_context{0, chunk_count, context, func};
  PTP_WORK work = CreateThreadpoolWork(__apply_win32_callback, &work_context, nullptr);
  if (!work) {
    // In case of failure, fall back to serial execution.
    __apply_serial(chunk_count, context, func);
    return;
  }

  // Submit the work `chunk_count-1` times to the thread pool.
  // The current calling thread will also participate, hence 1 less element is submitted.
  for (size_t i = 1; i < chunk_count; ++i)
    SubmitThreadpoolWork(work);

  // Participate in the work.
  __apply_win32_reentrant = true;
  for (size_t chunk; (chunk = work_context.iteration.fetch_add(1, std::memory_order_relaxed)) < chunk_count;)
    func(context, chunk);
  __apply_win32_reentrant = false;

  // Wait until all running callbacks have completed, cancelling any pending ones.
  WaitForThreadpoolWorkCallbacks(work, true);

  // Cleanup.
  CloseThreadpoolWork(work);
}
#endif

void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
#if defined(__APPLE__)
  // On Darwin platforms, use libdispatch to provide parallel execution.
  ::dispatch_apply_f(chunk_count, DISPATCH_APPLY_AUTO, context, func);
#elif defined(_WIN32)
  // On Windows platforms, use the thread pool API to provide parallel execution.
  __apply_win32(chunk_count, context, func);
#else
  // Currently there's no parallelism provider other platforms.
  // A basic task scheduler based on a thread pool will be added later to perform the application in parallel.
  __apply_serial(chunk_count, context, func);
#endif
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
