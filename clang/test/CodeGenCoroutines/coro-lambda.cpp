// Verify that we synthesized the coroutine for a lambda inside of a function template.
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s

namespace std {
template <typename R, typename... T> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <class Promise = void> struct coroutine_handle;
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
template <class Promise> struct coroutine_handle : coroutine_handle<void> {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};
} // namespace std

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct Task {
  struct promise_type {
    Task get_return_object();
    void return_void() {}
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void unhandled_exception() noexcept;
  };
};

template <typename _AwrT> auto SyncAwait(_AwrT &&A) {
  if (!A.await_ready()) {
    auto AwaitAsync = [&]() -> Task {
      try { (void)(co_await A); } catch (...) {}
    };
    Task t = AwaitAsync();
  }
  return A.await_resume();
}

void f() {
  suspend_always test;
  SyncAwait(test);
}

// Verify that we synthesized the coroutine for a lambda inside SyncAwait
// CHECK-LABEL: define linkonce_odr void @_ZZ9SyncAwaitIR14suspend_alwaysEDaOT_ENKUlvE_clEv(
//   CHECK: alloca %"struct.Task::promise_type"
//   CHECK: call token @llvm.coro.id(
//   CHECK: call i8 @llvm.coro.suspend(
//   CHECK: call void @llvm.coro.end(
