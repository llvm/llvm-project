//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <new>

#include "include/atomic_support.h"

#if defined(_LIBCPP_ABI_MICROSOFT)
#  define _LIBPCPP_DEFINE_NEW_HANDLER
#elif defined(LIBCXX_BUILDING_LIBCXXABI)
// nothing to do, we use the one from libc++abi
#elif defined(LIBCXXRT)
#  define _LIBPCPP_DEFINE_NEW_HANDLER
#elif defined(__GLIBCXX__)
// nothing to do, we use the one from libstdc++/libsupc++
#else
#  define _LIBPCPP_DEFINE_NEW_HANDLER
#endif

#if defined(_LIBPCPP_DEFINE_NEW_HANDLER)

static constinit std::new_handler __new_handler = nullptr;

#  ifdef _LIBCPP_ABI_VCRUNTIME
// to avoid including <new.h>
using _new_h = int(__cdecl*)(size_t);
extern "C" _new_h __cdecl _set_new_handler(_new_h);

namespace {
// adapter for _callnewh
int __cdecl _new_handler_adapter(size_t) {
  std::__libcpp_atomic_load (&__new_handler)();
  return 1;
}
} // namespace
#  endif

namespace std { // purposefully not versioned

new_handler set_new_handler(new_handler handler) noexcept {
#  ifdef _LIBCPP_ABI_VCRUNTIME
  auto old = __libcpp_atomic_exchange(&__new_handler, handler);
  _set_new_handler(handler ? _new_handler_adapter : nullptr);
  return old;
#  else
  return __libcpp_atomic_exchange(&__new_handler, handler);
#  endif
}

new_handler get_new_handler() noexcept { return __libcpp_atomic_load(&__new_handler); }

} // namespace std

#endif // _LIBPCPP_DEFINE_NEW_HANDLER
