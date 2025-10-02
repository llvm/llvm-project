// Test that we do not destroy the coroutine promise before coroutine completion, even if the coroutine never suspends.
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:            -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s --check-prefix=CHECK-O0
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:            -fexceptions -fcxx-exceptions -O2 | FileCheck %s --check-prefix=CHECK-O2

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

// CHECK-O0-LABEL: define {{.*}} void @_Z4funcv(
// CHECK-O0: coro.end:
// CHECK-O0-NEXT: %never.suspend = phi i1 [ false, %cleanup.cont16 ], [ true, %cleanup12 ], [ false, %after.coro.free ], [ false, %final.suspend ], [ false, %init.suspend ]
// CHECK-O0-NEXT: call void @llvm.coro.end
// CHECK-O0-NEXT: ptr @_ZNK4coroI1AE13RValueWrappercvOS0_Ev
// CHECK-O0-NEXT: call void @llvm.memcpy.p0.p0.i64
// CHECK-O0-NEXT: br i1 %never.suspend, label %coro.cleanup.ramp, label %coro.ret

// CHECK-O0: coro.cleanup.ramp:
// CHECK-O0-NEXT: call void @_ZN1AD1Ev
// CHECK-O0-NEXT: call void @llvm.lifetime.end.p0
// CHECK-O0-NEXT: call ptr @llvm.coro.free

// CHECK-O0: coro.ret:
// CHECK-O0-NEXT: call void @llvm.lifetime.end.p0
// CHECK-O0-NEXT: ret void

// CHECK-O2: store i32 5
