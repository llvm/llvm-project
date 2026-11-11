//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <__config>
#include <__mutex/once_flag.h>
#include <__pstl/backends/statpart.h>
#include <__utility/no_destroy.h>

#include "std_thread_sched.h"

#if defined(__APPLE__)
#  include <dispatch/dispatch.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
  _LIBCPP_ASSERT_NON_NULL(func != nullptr, "Attempting to apply a function which is a nullptr");
#if defined(__APPLE__)
  // On Apple platforms use libdispatch to provide the parallelism.
  ::dispatch_apply_f(chunk_count, DISPATCH_APPLY_AUTO, context, func);
#elif _LIBCPP_HAS_THREADS
  // Otherwise, if threading is enabled on the platform, use the task scheduler built on top of std::thread.
  // Scheduler is stored in __no_destroy<> instead of using _LIBCPP_NO_DESTROY to guarantee it's never destroyed
  // regardless of a compiler; not destroying is not an optimization, it's a matter of correctness here.
  static constinit std::once_flag once;
  static constinit std::__no_destroy<Scheduler> sched{std::__uninitialized_tag{}};
  std::call_once(once, []() { sched.__emplace(); });
  sched.__get().apply(chunk_count, context, func);
#else
  // In case there's no threading available on the platform, apply the functor serially.
  // Calls like these should not happen in practice, but we need to preserve the semantics regardless.
  for (std::size_t i = 0; i < chunk_count; ++i) {
    func(context, i);
  }
#endif
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
