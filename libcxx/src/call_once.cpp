//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__mutex/once_flag.h>
#include <__utility/exception_guard.h>

#if _LIBCPP_HAS_THREADS
#  include <__thread/support.h>
#endif

#include "include/atomic_support.h"

#include <__atomic/contention_t.h>

_LIBCPP_BEGIN_NAMESPACE_STD

// If dispatch_once_f ever handles C++ exceptions, and if one can get to it
// without illegal macros (unexpected macros not beginning with _UpperCase or
// __lowercase), and if it stops spinning waiting threads, then call_once should
// call into dispatch_once_f instead of here. Relevant radar this code needs to
// keep in sync with:  7741191.

#if _LIBCPP_HAS_THREADS
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_wait(void const volatile*, __cxx_contention_t) _NOEXCEPT;
#endif

void __call_once(volatile once_flag::_State_type& flag, void* arg, void (*func)(void*)) {
#if !_LIBCPP_HAS_THREADS

  if (flag == once_flag::_Unset) {
    auto guard = std::__make_exception_guard([&flag] { flag = once_flag::_Unset; });
    flag       = once_flag::_Pending;
    func(arg);
    flag = once_flag::_Complete;
    guard.__complete();
  }

#else // !_LIBCPP_HAS_THREADS

  auto flag_read = __atomic_load_n(&flag, __ATOMIC_ACQUIRE);

WAIT:
  while (flag_read == once_flag::_Pending) {
    __cxx_contention_t monitor = __libcpp_atomic_monitor(&flag);
    flag_read                  = __atomic_load_n(&flag, __ATOMIC_ACQUIRE);
    if (flag_read == once_flag::_Pending) {
      __libcpp_atomic_wait(&flag, monitor);
      flag_read = __atomic_load_n(&flag, __ATOMIC_ACQUIRE);
    }
  }

  if (flag_read == once_flag::_Unset) {
    once_flag::_State_type expected = once_flag::_Unset;
    if (__atomic_compare_exchange_n(&flag, &expected, once_flag::_Pending, false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE)) {
      auto guard = std::__make_exception_guard([&flag] {
        __libcpp_atomic_store(&flag, once_flag::_Unset, _AO_Release);
        __cxx_atomic_notify_all(&flag);
      });

      func(arg);

      __libcpp_atomic_store(&flag, once_flag::_Complete, _AO_Release);
      __cxx_atomic_notify_all(&flag);
      guard.__complete();

    } else {
      if (expected == once_flag::_Pending) {
        flag_read = expected;
        goto WAIT;
      }
    }
  }

#endif // !_LIBCPP_HAS_THREADS
}

_LIBCPP_END_NAMESPACE_STD
