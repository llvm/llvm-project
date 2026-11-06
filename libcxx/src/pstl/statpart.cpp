//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__pstl/backends/statpart.h>

#if defined(__APPLE__)
#  include <dispatch/dispatch.h>
#else
#  include "std_thread_sched.h"
#  include <__mutex/once_flag.h>
#  include <__utility/no_destroy.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
#if defined(__APPLE__)
  ::dispatch_apply_f(chunk_count, DISPATCH_APPLY_AUTO, context, func);
#else
  static constinit std::once_flag once;
  static constinit std::__no_destroy<Sched> sched{std::__uninitialized_tag{}};
  std::call_once(once, []() { sched.__emplace(); });
  sched.__get().apply(chunk_count, context, func);
#endif
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
