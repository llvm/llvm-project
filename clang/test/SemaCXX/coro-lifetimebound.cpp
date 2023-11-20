// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 -fsyntax-only -verify -Wall -Wextra -Wno-error=unreachable-code -Wno-unused

#include "Inputs/std-coroutine.h"

using std::suspend_always;
using std::suspend_never;

template <typename T> struct [[clang::coro_lifetimebound, clang::coro_return_type]] Gen {
  struct promise_type {
    Gen<T> get_return_object() {
      return {};
    }
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_value(const T &t);

    template <typename U>
    auto await_transform(const Gen<U> &) {
      struct awaitable {
        bool await_ready() noexcept { return false; }
        void await_suspend(std::coroutine_handle<>) noexcept {}
        U await_resume() noexcept { return {}; }
      };
      return awaitable{};
    }
  };
};

template <typename T> using Co = Gen<T>;

Gen<int> foo_coro(const int& b);

Gen<int> foo_coro(const int& b) {
  if (b > 0)
    co_return 1;
  co_return 2;
}

int getInt() { return 0; }

Co<int> bar_coro(const int &b, int c) {
  int x = co_await foo_coro(b);
  int y = co_await foo_coro(1);
  int z = co_await foo_coro(getInt());
  auto unsafe1 = foo_coro(1); // expected-warning {{temporary whose address is used as value of local variable}}
  auto unsafe2 = foo_coro(getInt()); // expected-warning {{temporary whose address is used as value of local variable}}
  auto  safe1 = foo_coro(b);
  auto  safe2 = foo_coro(c);
  co_return co_await foo_coro(co_await foo_coro(1));
}

[[clang::coro_wrapper]] Gen<int> plain_return_co(int b) {
  return foo_coro(b); // expected-warning {{address of stack memory associated with parameter}}
}

[[clang::coro_wrapper]] Gen<int> safe_forwarding(const int& b) {
  return foo_coro(b);
}

[[clang::coro_wrapper]] Gen<int> unsafe_wrapper(int b) {
  return safe_forwarding(b); // expected-warning {{address of stack memory associated with parameter}}
}

[[clang::coro_wrapper]] Co<int> complex_plain_return(int b) {
  return b > 0 
      ? foo_coro(1)   // expected-warning {{returning address of local temporary object}}
      : bar_coro(0, 1); // expected-warning {{returning address of local temporary object}}
}

#define CORO_WRAPPER \
  _Pragma("clang diagnostic push") \
  _Pragma("clang diagnostic ignored \"-Wc++23-extensions\"") \
  [[clang::coro_wrapper]] \
  _Pragma("clang diagnostic pop")

void lambdas() {
  auto unsafe_lambda = [] CORO_WRAPPER (int b) {
    return foo_coro(b); // expected-warning {{address of stack memory associated with parameter}}
  };
  auto safe_lambda = [](int b) -> Co<int> {
    int x = co_await foo_coro(1);
    co_return x + co_await foo_coro(b);
  };
}
// =============================================================================
// Safe usage when parameters are value
// =============================================================================
namespace by_value {
Gen<int> value_coro(int b) { co_return co_await foo_coro(b); }
[[clang::coro_wrapper]] Gen<int> wrapper1(int b) { return value_coro(b); }
[[clang::coro_wrapper]] Gen<int> wrapper2(const int& b) { return value_coro(b); }
}
