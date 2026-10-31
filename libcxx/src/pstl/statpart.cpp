//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__pstl/backends/statpart.h>

#if 0 // code from Main
#  if defined(__APPLE__)
#    include <dispatch/dispatch.h>
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
#  if defined(__APPLE__)
  ::dispatch_apply_f(chunk_count, DISPATCH_APPLY_AUTO, context, func);
#  else
  // Currently there's no parallelism provider on non-Apple platforms.
  // A basic task scheduler based on a thread pool will be added later to perform the application in parallel.
  for (size_t i = 0; i < chunk_count; ++i)
    func(context, i);
#  endif
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#else

#  include "std_thread_sched.h"
#  include <__mutex/once_flag.h>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

static constinit Sched* g_sched = nullptr;
static constinit std::once_flag g_sched_once_flag;

static Sched* get_sched() {
  std::call_once(g_sched_once_flag, []() { g_sched = new Sched(); });
  return g_sched;
}

void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
  if (chunk_count < 2) {
    apply_serial(chunk_count, context, func);
    return;
  }
  Sched* sched = get_sched();
  sched->apply(chunk_count, context, func);
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif
