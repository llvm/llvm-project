//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCPP___COROUTINE_TRIVIAL_AWAITABLES_H
#define __LIBCPP___COROUTINE_TRIVIAL_AWAITABLES_H

#include <__config>
#include <__coroutine/coroutine_handle.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

// [coroutine.trivial.awaitables]
struct suspend_never {
  constexpr bool await_ready() const noexcept { return true; }
  constexpr void await_suspend(coroutine_handle<>) const noexcept {}
  constexpr void await_resume() const noexcept {}
};

struct suspend_always {
  constexpr bool await_ready() const noexcept { return false; }
  constexpr void await_suspend(coroutine_handle<>) const noexcept {}
  constexpr void await_resume() const noexcept {}
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // __LIBCPP___COROUTINE_TRIVIAL_AWAITABLES_H
