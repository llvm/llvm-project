// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -stdlib=libc++ -std=c++1z -fcoroutines-ts -fsyntax-only

namespace std::experimental {
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
}

struct suspend_never {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct MoveOnly {
  MoveOnly() {};
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) noexcept {};
  ~MoveOnly() {};
};

template <typename T>
struct task {
  struct promise_type {
    auto initial_suspend() { return suspend_never{}; }
    auto final_suspend() noexcept { return suspend_never{}; }
    auto get_return_object() { return task{}; }
    static void unhandled_exception() {}
    void return_value(T&& value) {}
  };
};

task<MoveOnly> f() {
  MoveOnly value;
  co_return value;
}

int main() {
  f();
  return 0;
}

// expected-no-diagnostics
