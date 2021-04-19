// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fcoroutines-ts \
// RUN:   -fexperimental-new-pass-manager -O0 %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fcoroutines-ts \
// RUN:   -fexperimental-new-pass-manager -fno-inline -O0 %s -o - | FileCheck %s

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
// CHECK-NEXT: %this.addr.i{{[0-9]*}} = alloca %"struct.std::experimental::awaitable"*, align 8
// CHECK-NEXT: %this.addr.i{{[0-9]*}} = alloca %"struct.std::experimental::awaitable"*, align 8
// CHECK: [[CAST0:%[0-9]+]] = bitcast %"struct.std::experimental::awaitable"** %this.addr.i{{[0-9]*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[CAST0]])
// CHECK: [[CAST1:%[0-9]+]] = bitcast %"struct.std::experimental::awaitable"** %this.addr.i{{[0-9]*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[CAST1]])

// CHECK: [[CAST2:%[0-9]+]] = bitcast %"struct.std::experimental::awaitable"** %this.addr.i{{[0-9]*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[CAST2]])
// CHECK: [[CAST3:%[0-9]+]] = bitcast %"struct.std::experimental::awaitable"** %this.addr.i{{[0-9]*}} to i8*
// CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[CAST3]])
void foo() { co_return; }
