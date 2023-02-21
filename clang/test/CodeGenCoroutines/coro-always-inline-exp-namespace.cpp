// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fcoroutines-ts \
// RUN:   -O0 %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fcoroutines-ts \
// RUN:   -fno-inline -O0 %s -o - | FileCheck %s

namespace std {
namespace experimental {

struct handle {};

struct awaitable {
  bool await_ready() noexcept { return true; }
  // CHECK-NOT: await_suspend
  inline void __attribute__((__always_inline__)) await_suspend(handle) noexcept {}
  bool await_resume() noexcept { return true; }
};

template <typename T>
struct coroutine_handle {
  static handle from_address(void *address) noexcept { return {}; }
};

template <typename T = void>
struct coroutine_traits {
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

// CHECK-LABEL: @_Z3foov
// CHECK-LABEL: entry:
// CHECK: call void @llvm.lifetime.start.p0(i64 1, ptr %ref.tmp{{.*}})
// CHECK: call void @llvm.lifetime.end.p0(i64 1, ptr %ref.tmp{{.*}})

// CHECK: call void @llvm.lifetime.start.p0(i64 1, ptr %ref.tmp{{.*}})
// CHECK: call void @llvm.lifetime.end.p0(i64 1, ptr %ref.tmp{{.*}})
void foo() { co_return; }
