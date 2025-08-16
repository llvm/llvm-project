// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -disable-llvm-passes | FileCheck %s --check-prefix=CHECK-INITIAL
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -O2 | FileCheck %s --check-prefix=CHECK-OPTIMIZED

#include "Inputs/coroutine.h"

// Awaitable with `coro_await_suspend_destroy` attribute
struct [[clang::coro_await_suspend_destroy]] DestroyingAwaitable {
  bool await_ready() { return false; }
  void await_suspend_destroy(auto& promise) {}
  void await_suspend(auto handle) {
    await_suspend_destroy(handle.promise());
    handle.destroy();
  }
  void await_resume() {}
};

// Awaitable without `coro_await_suspend_destroy` (normal behavior)
struct NormalAwaitable {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<> h) {}
  void await_resume() {}
};

// Coroutine type with `std::suspend_never` for initial/final suspend
struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

// Single co_await with coro_await_suspend_destroy.
// Should result in no allocation after optimization.
Task test_single_destroying_await() {
  co_await DestroyingAwaitable{};
}

// CHECK-INITIAL-LABEL: define{{.*}} void @_Z28test_single_destroying_awaitv
// CHECK-INITIAL: call{{.*}} @llvm.coro.alloc
// CHECK-INITIAL: call{{.*}} @llvm.coro.begin

// CHECK-OPTIMIZED-LABEL: define{{.*}} void @_Z28test_single_destroying_awaitv
// CHECK-OPTIMIZED-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPTIMIZED-NOT: call{{.*}} malloc
// CHECK-OPTIMIZED-NOT: call{{.*}} @_Znwm

// Test multiple `co_await`s, all with `coro_await_suspend_destroy`.
// This should also result in no allocation after optimization.
Task test_multiple_destroying_awaits(bool condition) {
  co_await DestroyingAwaitable{};
  co_await DestroyingAwaitable{};
  if (condition) {
    co_await DestroyingAwaitable{};
  }
}

// CHECK-INITIAL-LABEL: define{{.*}} void @_Z31test_multiple_destroying_awaitsb
// CHECK-INITIAL: call{{.*}} @llvm.coro.alloc
// CHECK-INITIAL: call{{.*}} @llvm.coro.begin

// CHECK-OPTIMIZED-LABEL: define{{.*}} void @_Z31test_multiple_destroying_awaitsb
// CHECK-OPTIMIZED-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPTIMIZED-NOT: call{{.*}} malloc
// CHECK-OPTIMIZED-NOT: call{{.*}} @_Znwm

// Mixed awaits - some with `coro_await_suspend_destroy`, some without.
// We should still see allocation because not all awaits destroy the coroutine.
Task test_mixed_awaits() {
  co_await NormalAwaitable{}; // Must precede "destroy" to be reachable
  co_await DestroyingAwaitable{};
}

// CHECK-INITIAL-LABEL: define{{.*}} void @_Z17test_mixed_awaitsv
// CHECK-INITIAL: call{{.*}} @llvm.coro.alloc
// CHECK-INITIAL: call{{.*}} @llvm.coro.begin

// CHECK-OPTIMIZED-LABEL: define{{.*}} void @_Z17test_mixed_awaitsv
// CHECK-OPTIMIZED: call{{.*}} @_Znwm


// Check the attribute detection affects control flow.  
Task test_attribute_detection() {
  co_await DestroyingAwaitable{};
  // Unreachable in OPTIMIZED, so those builds don't see an allocation.
  co_await NormalAwaitable{};
}

// Check that we skip the normal suspend intrinsic and go directly to cleanup.
//
// CHECK-INITIAL-LABEL: define{{.*}} void @_Z24test_attribute_detectionv
// CHECK-INITIAL: call{{.*}} @_Z24test_attribute_detectionv.__await_suspend_wrapper__await
// CHECK-INITIAL-NEXT: br label %cleanup5
// CHECK-INITIAL-NOT: call{{.*}} @llvm.coro.suspend
// CHECK-INITIAL: call{{.*}} @_Z24test_attribute_detectionv.__await_suspend_wrapper__await
// CHECK-INITIAL: call{{.*}} @llvm.coro.suspend
// CHECK-INITIAL: call{{.*}} @_Z24test_attribute_detectionv.__await_suspend_wrapper__final

// Since `co_await DestroyingAwaitable{}` gets converted into an unconditional
// branch, the `co_await NormalAwaitable{}` is unreachable in optimized builds.
// 
// CHECK-OPTIMIZED-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPTIMIZED-NOT: call{{.*}} malloc
// CHECK-OPTIMIZED-NOT: call{{.*}} @_Znwm

// Template awaitable with `coro_await_suspend_destroy` attribute
template<typename T>
struct [[clang::coro_await_suspend_destroy]] TemplateDestroyingAwaitable {
  bool await_ready() { return false; }
  void await_suspend_destroy(auto& promise) {}
  void await_suspend(auto handle) {
    await_suspend_destroy(handle.promise());
    handle.destroy();
  }
  void await_resume() {}
};

Task test_template_destroying_await() {
  co_await TemplateDestroyingAwaitable<int>{};
}

// CHECK-OPTIMIZED-LABEL: define{{.*}} void @_Z30test_template_destroying_awaitv
// CHECK-OPTIMIZED-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPTIMIZED-NOT: call{{.*}} malloc
// CHECK-OPTIMIZED-NOT: call{{.*}} @_Znwm
