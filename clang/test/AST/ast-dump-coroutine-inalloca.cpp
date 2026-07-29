// RUN: %clang_cc1 -triple i686-pc-windows-msvc -std=c++20 -ast-dump -ast-dump-filter test -Wno-coroutines-unsupported-target %s | FileCheck %s

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
struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};
} // namespace std

struct Noisy {
  int val;
  Noisy(int v);
  Noisy(const Noisy&);
  Noisy(Noisy&&) noexcept;
  ~Noisy();
};

struct Awaiter {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  Noisy await_resume() noexcept;
};

struct task {
  struct promise_type {
    task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

void consume(Noisy x); // passed by inalloca

task test() {
  consume(co_await Awaiter{});
}

// CHECK:      CallExpr {{.*}} 'void'
// CHECK-NEXT: | |-ImplicitCastExpr {{.*}} 'void (*)(Noisy)' <FunctionToPointerDecay>
// CHECK-NEXT: | | `-DeclRefExpr {{.*}} 'void (Noisy)' lvalue Function {{.*}} 'consume'
// CHECK-NEXT: | `-CoroutineSuspendParameterBypassExpr {{.*}} 'Noisy'
// CHECK-NEXT: |   |-bypass_sub_expr: MaterializeTemporaryExpr {{.*}} 'Noisy' xvalue
// CHECK:          `-bypass_move_expr: CXXBindTemporaryExpr {{.*}} 'Noisy'
// CHECK-NEXT:       `-CXXConstructExpr {{.*}} 'Noisy' 'void (Noisy &&) __attribute__((thiscall)) noexcept' elidable
// CHECK-NEXT:         `-MaterializeTemporaryExpr {{.*}} 'Noisy' xvalue
// CHECK-NEXT:           `-CoawaitExpr {{.*}} 'Noisy'
