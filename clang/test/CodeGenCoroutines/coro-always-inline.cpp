// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -emit-llvm -std=c++20 \
// RUN:   -O0 %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -emit-llvm -std=c++20 \
// RUN:   -fno-inline -O0 %s -o - | FileCheck %s

namespace std {

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
} // namespace std

// CHECK-LABEL: @_Z3foov
// CHECK-LABEL: entry:
// CHECK: [[CAST0:%[0-9]+]] = bitcast %"struct.std::awaitable"* %ref.tmp{{.*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* [[CAST0]])
// CHECK: [[CAST1:%[0-9]+]] = bitcast %"struct.std::awaitable"* %ref.tmp{{.*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* [[CAST1]])

// CHECK: [[CAST2:%[0-9]+]] = bitcast %"struct.std::awaitable"* %ref.tmp{{.*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* [[CAST2]])
// CHECK: [[CAST3:%[0-9]+]] = bitcast %"struct.std::awaitable"* %ref.tmp{{.*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* [[CAST3]])
void foo() { co_return; }

// Check that bar is not inlined even it's marked as always_inline.

// CHECK-LABEL:   define {{.*}} void @_Z3bazv()
// CHECK:         call void @_Z3barv(
__attribute__((__always_inline__)) void bar() {
  co_return;
}
void baz() {
  bar();
  co_return;
}
