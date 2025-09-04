// Test that destruction of promise is deferred if coroutine completed without suspending
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

template<class T>
struct coro {
  struct RValueWrapper {
    T* p;

    operator T&&() const noexcept { return static_cast<T&&>(*p); }
  };

  using promise_type = T;

  T& getDerived() noexcept { return *static_cast<T*>(this); }

  auto get_return_object() noexcept { return RValueWrapper(&getDerived()); }
  std::suspend_never initial_suspend() noexcept { return {}; }
  std::suspend_never final_suspend() noexcept { return {}; }
  void return_value(T&& x) noexcept { getDerived() = static_cast<T&&>(x); }
  void unhandled_exception() {}
};

struct A : public coro<A> {
  int a;

  ~A() {}
};

A func() {
  A aa{};
  aa.a = 5;
  co_return static_cast<A&&>(aa);
}


// CHECK-LABEL: define {{.*}} void @_Z4funcv(
// CHECK: coro.end:
// CHECK-NEXT: %never.suspend = phi i1 [ false, %cleanup.cont16 ], [ true, %cleanup12 ], [ false, %after.coro.free ], [ false, %final.suspend ], [ false, %init.suspend ]
// CHECK-NEXT: call i1 @llvm.coro.end
// CHECK-NEXT: ptr @_ZNK4coroI1AE13RValueWrappercvOS0_Ev
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64
// CHECK-NEXT: br i1 %never.suspend, label %coro.cleanup.defer, label %coro.ret

// CHECK: coro.cleanup.defer:
// CHECK-NEXT: call void @_ZN1AD1Ev
// CHECK-NEXT: call void @llvm.lifetime.end.p0
// CHECK-NEXT: call ptr @llvm.coro.free

// CHECK: coro.ret:
// CHECK-NEXT: call void @llvm.lifetime.end.p0
// CHECK-NEXT: ret void
