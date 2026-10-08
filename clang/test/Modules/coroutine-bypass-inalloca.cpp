// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple i686-pc-windows-msvc -Wno-coroutines-unsupported-target \
// RUN:   -emit-reduced-module-interface -o %t/m.pcm %t/m.cppm
// RUN: %clang_cc1 -std=c++20 -triple i686-pc-windows-msvc -Wno-coroutines-unsupported-target \
// RUN:   -fmodule-file=m=%t/m.pcm -fsyntax-only -verify %t/use.cpp

//--- m.cppm
export module m;

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
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
} // namespace std

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

export struct task {
  struct promise_type {
    task get_return_object() { return {}; }
    suspend_never initial_suspend() { return {}; }
    suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

export struct Noisy {
  int val;
  Noisy(int v);
  Noisy(const Noisy&) = delete;
  Noisy(Noisy&& o) noexcept;
  ~Noisy();
};

export struct Awaiter {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  Noisy await_resume() noexcept;
};

export void consume_two(Noisy x, Noisy y);

export inline task my_task() {
  consume_two(co_await Awaiter{}, Noisy(42));
}

//--- use.cpp
// expected-no-diagnostics
import m;
void use() {
  my_task();
}
