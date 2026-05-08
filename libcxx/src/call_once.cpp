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

_LIBCPP_BEGIN_NAMESPACE_STD

// If dispatch_once_f ever handles C++ exceptions, and if one can get to it
// without illegal macros (unexpected macros not beginning with _UpperCase or
// __lowercase), and if it stops spinning waiting threads, then call_once should
// call into dispatch_once_f instead of here. Relevant radar this code needs to
// keep in sync with:  7741191.

void __call_once(atomic<once_flag::_State_type>& flag, void* arg, void (*func)(void*)) {
#if !_LIBCPP_HAS_THREADS

  if (flag == once_flag::_Unset) {
    auto guard = std::__make_exception_guard([&flag] { flag = once_flag::_Unset; });
    flag       = once_flag::_Pending;
    func(arg);
    flag = once_flag::_Complete;
    guard.__complete();
  }

#else // !_LIBCPP_HAS_THREADS

  auto flag_read = flag.load(memory_order_acquire);

WAIT:
  while (flag_read == once_flag::_Pending) {
    flag.wait(once_flag::_Pending, memory_order_acquire);
    flag_read = flag.load(memory_order_acquire);
  }

  if (flag_read == once_flag::_Unset) {
    once_flag::_State_type expected = once_flag::_Unset;
    if (flag.compare_exchange_strong(expected, once_flag::_Pending, memory_order_acquire, memory_order_acquire)) {
      auto guard = std::__make_exception_guard([&flag] {
        flag.store(once_flag::_Unset, memory_order_release);
        flag.notify_all();
      });

      func(arg);

      flag.store(once_flag::_Complete, memory_order_release);
      flag.notify_all();
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
