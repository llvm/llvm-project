// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 -fsyntax-only -verify -Wall -Wextra
#include "Inputs/std-coroutine.h"

using std::suspend_always;
using std::suspend_never;


template <typename T> struct [[clang::coro_return_type]] Gen {
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

Gen<int> foo_coro(int b);
Gen<int> foo_coro(int b) { co_return b; }

[[clang::coro_wrapper]] Gen<int> marked_wrapper1(int b) { return foo_coro(b); }

// expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
Gen<int> non_marked_wrapper(int b) { return foo_coro(b); }

namespace using_decl {
template <typename T> using Co = Gen<T>;

[[clang::coro_wrapper]] Co<int> marked_wrapper1(int b) { return foo_coro(b); }

// expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
Co<int> non_marked_wrapper(int b) { return foo_coro(b); }
} // namespace using_decl

namespace lambdas {
void foo() {
  auto coro_lambda = []() -> Gen<int> {
    co_return 1;
  };
  auto wrapper_lambda = [&]() -> Gen<int> {
    return coro_lambda();
  }
}
}