// RUN: %clang_cc1 -std=c++1z -fcoroutines-ts -verify %s -fcxx-exceptions -fexceptions -triple x86_64-windows-msvc -fms-extensions
namespace std::experimental {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
}

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <> struct std::experimental::coroutine_traits<void> {
  struct promise_type {
    void get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    void unhandled_exception() noexcept;
  };
};

void SEH_used() {
  __try { // expected-error {{cannot use SEH '__try' in a coroutine when C++ exceptions are enabled}}
    co_return; // expected-note {{function is a coroutine due to use of 'co_return' here}}
  } __except(0) {}
}
