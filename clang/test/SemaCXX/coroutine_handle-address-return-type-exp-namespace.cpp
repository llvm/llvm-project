// RUN: %clang_cc1 -verify %s -stdlib=libc++ -std=c++1z -fcoroutines-ts -fsyntax-only

namespace std::experimental {
template <class Promise = void>
struct coroutine_handle;

template <>
struct coroutine_handle<void> {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
  void *address() const;
};

template <class Promise>
struct coroutine_handle : public coroutine_handle<> {
};

template <class... Args>
struct void_t_imp {
  using type = void;
};
template <class... Args>
using void_t = typename void_t_imp<Args...>::type;

template <class T, class = void>
struct traits_sfinae_base {};

template <class T>
struct traits_sfinae_base<T, void_t<typename T::promise_type>> {
  using promise_type = typename T::promise_type;
};

template <class Ret, class... Args>
struct coroutine_traits : public traits_sfinae_base<Ret> {};
// expected-note@-1{{declared here}}
} // namespace std::experimental

struct suspend_never {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct task {
  struct promise_type {
    auto initial_suspend() { return suspend_never{}; }
    auto final_suspend() noexcept { return suspend_never{}; }
    auto get_return_object() { return task{}; }
    static void unhandled_exception() {}
    void return_void() {}
  };
};

namespace std::experimental {
template <>
struct coroutine_handle<task::promise_type> : public coroutine_handle<> {
  coroutine_handle<task::promise_type> *address() const; // expected-warning {{return type of 'coroutine_handle<>::address should be 'void*'}}
};
} // namespace std::experimental

struct awaitable {
  bool await_ready();

  std::experimental::coroutine_handle<task::promise_type>
  await_suspend(std::experimental::coroutine_handle<> handle);
  void await_resume();
} a;

task f() {
  co_await a; // expected-warning {{support for std::experimental::coroutine_traits will be removed}}
}

int main() {
  f();
  return 0;
}
