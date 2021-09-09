// This test merely verifies that emitting the object file does not cause a
// crash when the LLVM coroutines passes are run.
// PR42867: Disable this test for the new PM since the passes that lower the
// llvm.coro.* intrinsics have not yet been ported.
// RUN: %clang_cc1 -fno-experimental-new-pass-manager -emit-obj -std=c++2a -fsanitize=null %s -o %t.o

namespace std::experimental {
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
}

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::experimental::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct task {
  struct promise_type {
    task get_return_object() { return task(); }
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

struct awaitable {
  task await() { (void)co_await *this; }
  bool await_ready() { return false; }
  bool await_suspend(std::experimental::coroutine_handle<> awaiter) { return false; }
  bool await_resume() { return false; }
};

int main() {
  awaitable a;
  a.await();
}
