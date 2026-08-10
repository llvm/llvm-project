//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class C>
// explicit stop_callback(const stop_token& st, C&& cb)
//   noexcept(is_nothrow_constructible_v<Callback, C>);

#include <atomic>
#include <cassert>
#include <chrono>
#include <stop_token>
#include <type_traits>
#include <utility>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

struct Cb {
  void operator()() const;
};

// Constraints: Callback and C satisfy constructible_from<Callback, C>.
static_assert(std::is_constructible_v<std::stop_callback<void (*)()>, const std::stop_token&, void (*)()>);
static_assert(!std::is_constructible_v<std::stop_callback<void (*)()>, const std::stop_token&, void (*)(int)>);
static_assert(std::is_constructible_v<std::stop_callback<Cb>, const std::stop_token&, Cb&>);
static_assert(std::is_constructible_v<std::stop_callback<Cb&>, const std::stop_token&, Cb&>);
static_assert(!std::is_constructible_v<std::stop_callback<Cb>, const std::stop_token&, int>);

// explicit
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };
static_assert(ImplicitlyConstructible<int, int>);
static_assert(!ImplicitlyConstructible<std::stop_callback<Cb>, const std::stop_token&, Cb>);

// noexcept
template <bool NoExceptCtor>
struct CbNoExcept {
  CbNoExcept(int) noexcept(NoExceptCtor);
  void operator()() const;
};
static_assert(std::is_nothrow_constructible_v<std::stop_callback<CbNoExcept<true>>, const std::stop_token&, int>);
static_assert(!std::is_nothrow_constructible_v<std::stop_callback<CbNoExcept<false>>, const std::stop_token&, int>);

int main(int, char**) {
  // was requested
  {
    std::stop_source ss;
    const std::stop_token st = ss.get_token();
    ss.request_stop();

    bool called = false;
    std::stop_callback sc(st, [&] { called = true; });
    assert(called);
  }

  // was not requested
  {
    std::stop_source ss;
    const std::stop_token st = ss.get_token();

    bool called = false;
    std::stop_callback sc(st, [&] { called = true; });
    assert(!called);

    ss.request_stop();
    assert(called);
  }

  // token has no state
  {
    std::stop_token st;
    bool called = false;
    std::stop_callback sc(st, [&] { called = true; });
    assert(!called);
  }

  // should not be called multiple times
  {
    std::stop_source ss;
    const std::stop_token st = ss.get_token();

    int calledTimes = 0;
    std::stop_callback sc(st, [&] { ++calledTimes; });

    std::vector<std::thread> threads;
    for (auto i = 0; i < 10; ++i) {
      threads.emplace_back(support::make_test_thread([&] { ss.request_stop(); }));
    }

    for (auto& thread : threads) {
      thread.join();
    }
    assert(calledTimes == 1);
  }

  // adding more callbacks during invoking other callbacks
  {
    std::stop_source ss;
    const std::stop_token st = ss.get_token();

    std::atomic<bool> startedFlag = false;
    std::atomic<bool> finishFlag  = false;
    std::stop_callback sc(st, [&] {
      startedFlag = true;
      startedFlag.notify_all();
      finishFlag.wait(false);
    });

    auto thread = support::make_test_thread([&] { ss.request_stop(); });

    startedFlag.wait(false);

    // first callback is still running, adding another one;
    bool secondCallbackCalled = false;
    std::stop_callback sc2(st, [&] { secondCallbackCalled = true; });

    finishFlag = true;
    finishFlag.notify_all();

    thread.join();
    assert(secondCallbackCalled);
  }

  // adding callbacks on different threads
  {
    std::stop_source ss;
    const std::stop_token st = ss.get_token();

    std::vector<std::thread> threads;
    std::atomic<int> callbackCalledTimes = 0;
    std::atomic<bool> done               = false;
    for (auto i = 0; i < 10; ++i) {
      threads.emplace_back(support::make_test_thread([&] {
        std::stop_callback sc{st, [&] { callbackCalledTimes.fetch_add(1, std::memory_order_relaxed); }};
        done.wait(false);
      }));
    }
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1ms);
    ss.request_stop();
    done = true;
    done.notify_all();
    for (auto& thread : threads) {
      thread.join();
    }
    assert(callbackCalledTimes.load(std::memory_order_relaxed) == 10);
  }

  // correct overload
  {
    struct CBWithTracking {
      bool& lvalueCalled;
      bool& lvalueConstCalled;
      bool& rvalueCalled;
      bool& rvalueConstCalled;

      void operator()() & { lvalueCalled = true; }
      void operator()() const& { lvalueConstCalled = true; }
      void operator()() && { rvalueCalled = true; }
      void operator()() const&& { rvalueConstCalled = true; }
    };

    // RValue
    {
      bool lvalueCalled      = false;
      bool lvalueConstCalled = false;
      bool rvalueCalled      = false;
      bool rvalueConstCalled = false;
      std::stop_source ss;
      const std::stop_token st = ss.get_token();
      ss.request_stop();

      std::stop_callback<CBWithTracking> sc(
          st, CBWithTracking{lvalueCalled, lvalueConstCalled, rvalueCalled, rvalueConstCalled});
      assert(rvalueCalled);
    }

    // RValue
    {
      bool lvalueCalled      = false;
      bool lvalueConstCalled = false;
      bool rvalueCalled      = false;
      bool rvalueConstCalled = false;
      std::stop_source ss;
      const std::stop_token st = ss.get_token();
      ss.request_stop();

      std::stop_callback<const CBWithTracking> sc(
          st, CBWithTracking{lvalueCalled, lvalueConstCalled, rvalueCalled, rvalueConstCalled});
      assert(rvalueConstCalled);
    }

    // LValue
    {
      bool lvalueCalled      = false;
      bool lvalueConstCalled = false;
      bool rvalueCalled      = false;
      bool rvalueConstCalled = false;
      std::stop_source ss;
      const std::stop_token st = ss.get_token();
      ss.request_stop();
      CBWithTracking cb{lvalueCalled, lvalueConstCalled, rvalueCalled, rvalueConstCalled};
      std::stop_callback<CBWithTracking&> sc(st, cb);
      assert(lvalueCalled);
    }

    // const LValue
    {
      bool lvalueCalled      = false;
      bool lvalueConstCalled = false;
      bool rvalueCalled      = false;
      bool rvalueConstCalled = false;
      std::stop_source ss;
      const std::stop_token st = ss.get_token();
      ss.request_stop();
      CBWithTracking cb{lvalueCalled, lvalueConstCalled, rvalueCalled, rvalueConstCalled};
      std::stop_callback<const CBWithTracking&> sc(st, cb);
      assert(lvalueConstCalled);
    }
  }

  return 0;
}
