// Tests that coroutine passes are added to and run by the new pass manager
// pipeline, at -O0 and above.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null \
// RUN:   -fdebug-pass-manager -std=c++20 \
// RUN:   -O0 %s 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null \
// RUN:   -fdebug-pass-manager -std=c++20 \
// RUN:   -O1 %s 2>&1 | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-OPT
//
// CHECK-ALL: Running pass:{{.*}}CoroEarlyPass
//
// CHECK-ALL: Running pass: CoroSplitPass on (_Z3foov)
// CHECK-OPT: Running pass:{{.*}}CoroElidePass{{.*}} on {{.*}}_Z3foov{{.*}}
//
// CHECK-ALL: Running pass:{{.*}}CoroCleanupPass

namespace std {

struct handle {};

struct awaitable {
  bool await_ready() noexcept { return false; }
  void await_suspend(handle) noexcept {}
  bool await_resume() noexcept { return true; }
};

template <typename T> struct coroutine_handle {
  static handle from_address(void *address) noexcept { return {}; }
};

template <typename T = void> struct coroutine_traits {
  struct promise_type {
    awaitable initial_suspend() { return {}; }
    awaitable final_suspend() noexcept { return {}; }
    void return_void() {}
    T get_return_object() { return T(); }
    void unhandled_exception() {}
  };
};
} // namespace std

void foo() { co_return; }
