// This file contains references to sections of the Coroutines TS, which can be
// found at http://wg21.link/coroutines.

// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -verify %s -fcxx-exceptions -fexceptions -Wunused-result

namespace std {
namespace experimental {
template <class Ret, typename... T>
struct coroutine_traits { using promise_type = typename Ret::promise_type; };
// expected-note@-1{{declared here}}

template <class Promise = void>
struct coroutine_handle {
  static coroutine_handle from_address(void *);
  void *address() const noexcept;
};
template <>
struct coroutine_handle<void> {
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>);
  void *address() const noexcept;
};

struct suspend_always {
  bool await_ready() { return false; }      // expected-note 2 {{must be declared with 'noexcept'}}
  void await_suspend(coroutine_handle<>) {} // expected-note 2 {{must be declared with 'noexcept'}}
  void await_resume() {}                    // expected-note 2 {{must be declared with 'noexcept'}}
  ~suspend_always() noexcept(false);        // expected-note 2 {{must be declared with 'noexcept'}}
};
} // namespace experimental
} // namespace std

using namespace std::experimental;

struct A {
  bool await_ready();
  void await_resume();
  template <typename F>
  void await_suspend(F);
};

struct coro_t {
  struct promise_type {
    coro_t get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend(); // expected-note 2 {{must be declared with 'noexcept'}}
    void return_void();
    static void unhandled_exception();
  };
};

coro_t f(int n) { // expected-error {{the expression 'co_await __promise.final_suspend()' is required to be non-throwing}}
  A a{};
  co_await a; // expected-warning {{support for 'std::experimental::coroutine_traits' will be removed}}
}

template <typename T>
coro_t f_dep(T n) { // expected-error {{the expression 'co_await __promise.final_suspend()' is required to be non-throwing}}
  A a{};
  co_await a;
}

void foo() {
  f_dep<int>(5); // expected-note {{in instantiation of function template specialization 'f_dep<int>' requested here}}
}

struct PositiveFinalSuspend {
  bool await_ready() noexcept;
  coroutine_handle<> await_suspend(coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct correct_coro {
  struct promise_type {
    correct_coro get_return_object();
    suspend_always initial_suspend();
    PositiveFinalSuspend final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
  };
};

correct_coro f2(int n) {
  co_return;
}

struct NegativeFinalSuspend {
  bool await_ready() noexcept;
  coroutine_handle<> await_suspend(coroutine_handle<>) noexcept;
  void await_resume() noexcept;
  ~NegativeFinalSuspend() noexcept(false); // expected-note {{must be declared with 'noexcept'}}
};

struct incorrect_coro {
  struct promise_type {
    incorrect_coro get_return_object();
    suspend_always initial_suspend();
    NegativeFinalSuspend final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
  };
};

incorrect_coro f3(int n) { // expected-error {{the expression 'co_await __promise.final_suspend()' is required to be non-throwing}}
  co_return;
}
