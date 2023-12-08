// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -std=c++20 %s

// expected-no-diagnostics

namespace std {
template <typename _Result, typename...>
struct coroutine_traits {
  using promise_type = typename _Result::promise_type;
};

template <typename _Promise = void>
struct coroutine_handle;

template <>
struct coroutine_handle<void> {
  static coroutine_handle from_address(void *__a) noexcept;
  void resume() const noexcept;
  void destroy() const noexcept;
};

template <typename _Promise>
struct coroutine_handle : coroutine_handle<> {};

struct suspend_always {
  bool await_ready() const noexcept;
  void await_suspend(coroutine_handle<>) const noexcept;
  void await_resume() const noexcept;
};
} // namespace std

class Task {
public:
  struct promise_type {
  public:
    std::suspend_always initial_suspend() noexcept;
    std::suspend_always final_suspend() noexcept;

    Task get_return_object() noexcept;
    void unhandled_exception() noexcept;
    void return_value(int value) noexcept;

    std::suspend_always yield_value(int value) noexcept;
  };
};

Task Foo() noexcept {
  // ICE'd
  co_yield({ int frame = 0; 0; });
  co_await({ int frame = 0; std::suspend_always(); });
  co_return({ int frame = 0; 0; });
}
