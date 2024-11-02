// RUN: %clang_cc1 -emit-llvm -o - -fcxx-exceptions -fexceptions -triple x86_64-apple-darwin10 %s | FileCheck %s

struct X {
  X();
  ~X();
};

struct Y { };

// CHECK-LABEL: define{{.*}} void @_Z1fv
// CHECK-SAME:  personality ptr @__gxx_personality_v0
void f() {
  // CHECK: call i32 @__cxa_guard_acquire(ptr @_ZGVZ1fvE1x)
  // CHECK: invoke void @_ZN1XC1Ev
  // CHECK: call i32 @__cxa_atexit
  // CHECK-NEXT: call void @__cxa_guard_release(ptr @_ZGVZ1fvE1x)
  // CHECK: br
  static X x;

  // CHECK: call ptr @__cxa_allocate_exception
  // CHECK: call void @__cxa_throw
  throw Y();

  // Finally, the landing pad.
  // CHECK: landingpad { ptr, i32 }
  // CHECK:   cleanup
  // CHECK: call void @__cxa_guard_abort(ptr @_ZGVZ1fvE1x)
  // CHECK: resume { ptr, i32 }
}
