// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// UNSUPPORTED: ubsan

// <experimental/coroutine>
// struct noop_coroutine_promise;
// using noop_coroutine_handle = coroutine_handle<noop_coroutine_promise>;
// noop_coroutine_handle noop_coroutine() noexcept;

#include <experimental/coroutine>
#include <cassert>
#include <type_traits>

#if __has_builtin(__builtin_coro_noop)

namespace coro = std::experimental::coroutines_v1;


static_assert(std::is_same<coro::coroutine_handle<coro::noop_coroutine_promise>, coro::noop_coroutine_handle>::value, "");
static_assert(std::is_same<decltype(coro::noop_coroutine()), coro::noop_coroutine_handle>::value, "");

// template <> struct coroutine_handle<noop_coroutine_promise> : coroutine_handle<>
// {
// // 18.11.2.7 noop observers
// constexpr explicit operator bool() const noexcept;
// constexpr bool done() const noexcept;

// // 18.11.2.8 noop resumption
// constexpr void operator()() const noexcept;
// constexpr void resume() const noexcept;
// constexpr void destroy() const noexcept;

// // 18.11.2.9 noop promise access
// noop_coroutine_promise& promise() const noexcept;

// // 18.11.2.10 noop address
// constexpr void* address() const noexcept;

int main(int, char**)
{
  auto h = coro::noop_coroutine();
  coro::coroutine_handle<> base = h;

  assert(h);
  assert(base);

  assert(!h.done());
  assert(!base.done());

  h.resume();
  h.destroy();
  h();
  static_assert(h.done() == false, "");
  static_assert(h, "");

  h.promise();
  assert(h.address() == base.address());
  assert(h.address() != nullptr);
  assert(coro::coroutine_handle<>::from_address(h.address()) == base);

  return 0;
}

#else

int main(int, char**) { return 0; }

#endif //  __has_builtin(__builtin_coro_noop)
