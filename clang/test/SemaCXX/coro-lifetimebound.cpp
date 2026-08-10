// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 -fsyntax-only -verify -Wall -Wextra -Wno-error=unreachable-code -Wno-unused -Wno-c++23-lambda-attributes

#include "Inputs/std-coroutine.h"

using std::suspend_always;
using std::suspend_never;

template <typename T> struct [[clang::coro_lifetimebound, clang::coro_return_type]] Co {
  struct promise_type {
    Co<T> get_return_object() {
      return {};
    }
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_value(const T &t);

    template <typename U>
    auto await_transform(const Co<U> &) {
      struct awaitable {
        bool await_ready() noexcept { return false; }
        void await_suspend(std::coroutine_handle<>) noexcept {}
        U await_resume() noexcept { return {}; }
      };
      return awaitable{};
    }
  };
};

Co<int> foo_coro(const int& b) {
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

[[clang::coro_wrapper]] Co<int> plain_return_co(int b) {
  return foo_coro(b); // expected-warning {{address of stack memory associated with parameter}}
}

[[clang::coro_wrapper]] Co<int> safe_forwarding(const int& b) {
  return foo_coro(b);
}

[[clang::coro_wrapper]] Co<int> unsafe_wrapper(int b) {
  return safe_forwarding(b); // expected-warning {{address of stack memory associated with parameter}}
}

[[clang::coro_wrapper]] Co<int> complex_plain_return(int b) {
  return b > 0 
      ? foo_coro(1)   // expected-warning {{returning address of local temporary object}}
      : bar_coro(0, 1); // expected-warning {{returning address of local temporary object}}
}

// =============================================================================
// Lambdas
// =============================================================================
namespace lambdas {
void lambdas() {
  auto unsafe_lambda = [] [[clang::coro_wrapper]] (int b) {
    return foo_coro(b); // expected-warning {{address of stack memory associated with parameter}}
  };
  auto coro_lambda = [] (const int&) -> Co<int> {
    co_return 0;
  };
  auto unsafe_coro_lambda = [&] (const int& b) -> Co<int> {
    int x = co_await coro_lambda(b);
    auto safe = coro_lambda(b);
    auto unsafe1 = coro_lambda(1); // expected-warning {{temporary whose address is used as value of local variable}}
    auto unsafe2 = coro_lambda(getInt()); // expected-warning {{temporary whose address is used as value of local variable}}
    auto unsafe3 = coro_lambda(co_await coro_lambda(b)); // expected-warning {{temporary whose address is used as value of local variable}}
    co_return co_await safe;
  };
  auto safe_lambda = [](int b) -> Co<int> {
    int x = co_await foo_coro(1);
    co_return x + co_await foo_coro(b);
  };
}

Co<int> lambda_captures() {
  int a = 1;
  // Temporary lambda object dies.
  auto lamb = [a](int x, const int& y) -> Co<int> { // expected-warning {{temporary whose address is used as value of local variable 'lamb'}}
    co_return x + y + a;
  }(1, a);
  // Object dies but it has no capture.
  auto no_capture = []() -> Co<int> { co_return 1; }();
  auto bad_no_capture = [](const int& a) -> Co<int> { co_return a; }(1); // expected-warning {{temporary}}
  // Temporary lambda object with lifetime extension under co_await.
  int res = co_await [a](int x, const int& y) -> Co<int> {
    co_return x + y + a;
  }(1, a);
  // Lambda object on stack should be fine.
  auto lamb2 = [a]() -> Co<int> { co_return a; };
  auto on_stack = lamb2();
  auto res2 = co_await on_stack;
  co_return 1;
}
} // namespace lambdas

// =============================================================================
// Member coroutines
// =============================================================================
namespace member_coroutines{
struct S {
  Co<int> member(const int& a) { co_return a; }
};

Co<int> use() {
  S s;
  int a = 1;
  auto test1 = s.member(1);  // expected-warning {{temporary whose address is used as value of local variable}}
  auto test2 = s.member(a);
  auto test3 = S{}.member(a);  // expected-warning {{temporary whose address is used as value of local variable}}
  co_return 1;
}

[[clang::coro_wrapper]] Co<int> wrapper(const int& a) {
  S s;
  return s.member(a); // expected-warning {{address of stack memory}}
}
} // member_coroutines

// =============================================================================
// Safe usage when parameters are value
// =============================================================================
namespace by_value {
Co<int> value_coro(int b) { co_return co_await foo_coro(b); }
[[clang::coro_wrapper]] Co<int> wrapper1(int b) { return value_coro(b); }
[[clang::coro_wrapper]] Co<int> wrapper2(const int& b) { return value_coro(b); }
} // namespace by_value

// =============================================================================
// Lifetime bound but not a Coroutine Return Type: No analysis.
// =============================================================================
namespace not_a_crt {
template <typename T> struct [[clang::coro_lifetimebound]] CoNoCRT {
  struct promise_type {
    CoNoCRT<T> get_return_object() {
      return {};
    }
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_value(const T &t);
  };
};

CoNoCRT<int> foo_coro(const int& a) { co_return a; }
CoNoCRT<int> bar(int a) { 
  auto x = foo_coro(a);
  co_return 1;
}
} // namespace not_a_crt

// =============================================================================
// Not lifetime bound coroutine wrappers: [[clang::coro_disable_lifetimebound]].
// =============================================================================
namespace disable_lifetimebound {
Co<int> foo(int x) {  co_return x; }

[[clang::coro_wrapper, clang::coro_disable_lifetimebound]]
Co<int> foo_wrapper(const int& x) { return foo(x); }

[[clang::coro_wrapper]] Co<int> caller() {
  // The call to foo_wrapper is wrapper is safe.
  return foo_wrapper(1);
}

struct S{
[[clang::coro_wrapper, clang::coro_disable_lifetimebound]]
Co<int> member(const int& x) { return foo(x); }
};

Co<int> use() {
  S s;
  int a = 1;
  auto test1 = s.member(1); // param is not flagged.
  auto test2 = S{}.member(a); // 'this' is not flagged.
  co_return 1;
}

[[clang::coro_wrapper]] Co<int> return_stack_addr(const int& a) {
  S s;
  return s.member(a); // return of stack addr is not flagged.
}
} // namespace disable_lifetimebound
