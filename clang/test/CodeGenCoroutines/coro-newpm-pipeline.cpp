// Tests that coroutine passes are added to and run by the new pass manager
// pipeline, at -O0 and above.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null \
// RUN:   -fexperimental-new-pass-manager -fdebug-pass-manager -fcoroutines-ts \
// RUN:   -O0 %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null \
// RUN:   -fexperimental-new-pass-manager -fdebug-pass-manager -fcoroutines-ts \
// RUN:   -O1 %s 2>&1 | FileCheck %s
//
// CHECK: Starting llvm::Module pass manager run.
// CHECK: Running pass:{{.*}}CoroEarlyPass
//
// The first coro-split pass enqueues a second run of the entire CGSCC pipeline.
// CHECK: Starting CGSCC pass manager run.
// CHECK: Running pass: CoroSplitPass on (_Z3foov)
// CHECK: Running pass:{{.*}}CoroElidePass{{.*}} on {{.*}}_Z3foov{{.*}}
// CHECK: Finished CGSCC pass manager run.
//
// The second coro-split pass splits coroutine 'foo' into funclets
// 'foo.resume', 'foo.destroy', and 'foo.cleanup'.
// CHECK: Starting CGSCC pass manager run.
// CHECK: Running pass: CoroSplitPass on (_Z3foov)
// CHECK: Running pass:{{.*}}CoroElidePass{{.*}} on {{.*}}_Z3foov{{.*}}
// CHECK: Finished CGSCC pass manager run.
//
// CHECK: Running pass:{{.*}}CoroCleanupPass
// CHECK: Finished llvm::Module pass manager run.

namespace std {
namespace experimental {

struct handle {};

struct awaitable {
  bool await_ready() noexcept { return true; }
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
} // namespace experimental
} // namespace std

void foo() { co_return; }
