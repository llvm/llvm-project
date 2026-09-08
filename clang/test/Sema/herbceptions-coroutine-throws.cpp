// RUN: %clang_cc1 -std=c++20 -fherbceptions -fsyntax-only -verify %s

// Coroutines must not use 'throws': all herbceptions must be caught within
// the coroutine body.

namespace std {
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
struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};
} // namespace std

struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

template <>
struct std::coroutine_traits<Task, int> {
  using promise_type = Task::promise_type;
};

// expected-error@+2 {{'throws' is not allowed on coroutine functions; all herbceptions must be caught within the coroutine body}}
Task make_task(int x) throws {
  co_return;
}
