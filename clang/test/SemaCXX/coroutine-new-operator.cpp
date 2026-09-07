// RUN: %clang_cc1 -std=c++20 -fcoro-aligned-allocation -fsyntax-only -verify %s
// expected-no-diagnostics

namespace std {
  template <typename R, typename... Args>
  struct coroutine_traits {
    using promise_type = typename R::promise_type;
  };

  template <class Promise = void> struct coroutine_handle {
    coroutine_handle() = default;
    static coroutine_handle from_address(void *) noexcept;
  };
  template <> struct coroutine_handle<void> {
    static coroutine_handle from_address(void *) noexcept;
    coroutine_handle() = default;
    template <class Promise>
    coroutine_handle(coroutine_handle<Promise>) noexcept;
  };

  struct suspend_always {
    bool await_ready() const noexcept { return false; }
    void await_suspend(coroutine_handle<>) const noexcept {}
    void await_resume() const noexcept {}
  };

  enum class align_val_t : decltype(sizeof(0)) {};
} // namespace std

using size_t = decltype(sizeof(0));

struct Task {
  struct promise_type {
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    Task get_return_object() noexcept { return {}; }

    void* operator new(size_t n, void* buf, size_t capacity) noexcept {
      return buf;
    }
    void operator delete(void*) noexcept {}
  };
};

Task CoGetReturnAddress(void* buf, size_t capacity) {
  co_return;
}
