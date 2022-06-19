//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: ubsan
// UNSUPPORTED: libcpp-no-coroutines

// <coroutine>
// struct noop_coroutine_promise;
// using noop_coroutine_handle = coroutine_handle<noop_coroutine_promise>;
// noop_coroutine_handle noop_coroutine() noexcept;

#include <coroutine>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_same<std::coroutine_handle<std::noop_coroutine_promise>, std::noop_coroutine_handle>::value, "");
static_assert(std::is_same<decltype(std::noop_coroutine()), std::noop_coroutine_handle>::value, "");

// template <> struct coroutine_handle<noop_coroutine_promise> : coroutine_handle<>
// {
// // [coroutine.handle.noop.observers], observers
// constexpr explicit operator bool() const noexcept;
// constexpr bool done() const noexcept;

// // [coroutine.handle.noop.resumption], resumption
// constexpr void operator()() const noexcept;
// constexpr void resume() const noexcept;
// constexpr void destroy() const noexcept;

// // [coroutine.handle.noop.promise], promise access
// noop_coroutine_promise& promise() const noexcept;

// // [coroutine.handle.noop.address], address
// constexpr void* address() const noexcept;

int main(int, char**)
{
  auto h = std::noop_coroutine();
  std::coroutine_handle<> base = h;

  assert(h);
  assert(base);

  assert(!h.done());
  assert(!base.done());

  h.resume();
  h.destroy();
  h();
  static_assert(h, "");
  static_assert(h.done() == false, "");

  // [coroutine.handle.noop.resumption]p2
  // Remarks: If noop_coroutine_handle is converted to
  // coroutine_handle<>, calls to operator(), resume and
  // destroy on that handle will also have no observable
  // effects.
  base.resume();
  base.destroy();
  base();
  assert(base);
  assert(base.done() == false);

  TEST_IGNORE_NODISCARD h.promise();
  assert(h.address() == base.address());
  assert(h == base);
  assert(h.address() != nullptr);
  assert(std::coroutine_handle<>::from_address(h.address()) == base);

  return 0;
}
