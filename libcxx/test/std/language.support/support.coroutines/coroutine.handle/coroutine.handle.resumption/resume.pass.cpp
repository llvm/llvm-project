//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-coroutines

// <coroutine>

// template <class Promise = void>
// struct coroutine_handle;

// void operator()() const
// void resume() const

#include <coroutine>
#include <type_traits>
#include <memory>
#include <utility>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

template <class H>
auto has_resume_imp(H&& h, int) -> decltype(h.resume(), std::true_type{});
template <class H>
auto has_resume_imp(H&&, long) -> std::false_type;

template <class H>
constexpr bool has_resume() {
  return decltype(has_resume_imp(std::declval<H>(), 0))::value;
}


template <class H>
auto has_call_operator_imp(H&& h, int) -> decltype(h(), std::true_type{});
template <class H>
auto has_call_operator_imp(H&&, long) -> std::false_type;

template <class H>
constexpr bool has_call_operator() {
  return decltype(has_call_operator_imp(std::declval<H>(), 0))::value;
}

template <class Promise>
void do_test(std::coroutine_handle<Promise>&& H) {
  using HType = std::coroutine_handle<Promise>;
  // FIXME Add a runtime test
  {
    ASSERT_SAME_TYPE(decltype(H.resume()), void);
    ASSERT_SAME_TYPE(decltype(H()), void);
    LIBCPP_ASSERT_NOT_NOEXCEPT(H.resume());
    LIBCPP_ASSERT_NOT_NOEXCEPT(H());
    static_assert(has_resume<HType&>(), "");
    static_assert(has_resume<HType&&>(), "");
    static_assert(has_call_operator<HType&>(), "");
    static_assert(has_call_operator<HType&&>(), "");
  }
  {
    static_assert(has_resume<HType const&>(), "");
    static_assert(has_resume<HType const&&>(), "");
    static_assert(has_call_operator<HType const&>(), "");
    static_assert(has_call_operator<HType const&&>(), "");
  }
}

int main(int, char**)
{
  do_test(std::coroutine_handle<>{});
  do_test(std::coroutine_handle<int>{});

  return 0;
}
