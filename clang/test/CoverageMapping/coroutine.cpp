// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping %s -o - | FileCheck %s

namespace std {
template <typename... T>
struct coroutine_traits;

template <class Promise = void>
struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept { return {}; }
};
template <>
struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) { return {}; }
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept {}
};
} // namespace std

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <>
struct std::coroutine_traits<int, int> {
  struct promise_type {
    int get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception() noexcept;
    void return_value(int);
    suspend_always yield_value(int);
  };
};

// CHECK-LABEL: _Z2f1i:
int f1(int x) {       // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE+8]]:2 = #0
  if (x > 42) {       // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:13 = #0
                      // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:13 = #1, (#0 - #1)
    ++x;              // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:14 -> [[@LINE-2]]:15 = #1
  } else {            // CHECK-NEXT: File 0, [[@LINE-3]]:15 -> [[@LINE]]:4 = #1
    co_return x + 42; // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:4 -> [[@LINE-1]]:10 = (#0 - #1)
  }                   // CHECK-NEXT: File 0, [[@LINE-2]]:10 -> [[@LINE]]:4 = (#0 - #1)
  co_return x;        // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:4 -> [[@LINE]]:3 = #1
} // CHECK-NEXT: File 0, [[@LINE-1]]:3 -> [[@LINE-1]]:14 = #1

// CHECK-LABEL: _Z2f2i:
// CHECK-NEXT: File 0, [[@LINE+1]]:15 -> [[@LINE+15]]:2 = #0
int f2(int x) {
// CHECK-NEXT: File 0, [[@LINE+5]]:13 -> [[@LINE+5]]:18 = #0
// CHECK-NEXT: Branch,File 0, [[@LINE+4]]:13 -> [[@LINE+4]]:18 = #1, (#0 - #1)
// CHECK-NEXT: Gap,File 0, [[@LINE+3]]:20 -> [[@LINE+3]]:21 = #1
// CHECK-NEXT: File 0, [[@LINE+2]]:21 -> [[@LINE+2]]:37 = #1
// CHECK-NEXT: File 0, [[@LINE+1]]:40 -> [[@LINE+1]]:56 = (#0 - #1)
  co_await (x > 0 ? suspend_always{} : suspend_always{});
// CHECK-NEXT: File 0, [[@LINE+5]]:12 -> [[@LINE+5]]:17 = #0
// CHECK-NEXT: Branch,File 0, [[@LINE+4]]:12 -> [[@LINE+4]]:17 = #2, (#0 - #2)
// CHECK-NEXT: Gap,File 0, [[@LINE+3]]:19 -> [[@LINE+3]]:20 = #2
// CHECK-NEXT: File 0, [[@LINE+2]]:20 -> [[@LINE+2]]:21 = #2
// CHECK-NEXT: File 0, [[@LINE+1]]:24 -> [[@LINE+1]]:25 = (#0 - #2)
  co_yield x > 0 ? 1 : 2;
  co_return 0;
}
