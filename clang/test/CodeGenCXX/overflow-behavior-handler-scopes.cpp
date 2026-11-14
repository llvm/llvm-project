// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fexperimental-overflow-behavior-types -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s

// Checks only reach the handler from code written in the function body.

int __ob_trap G;

// A default argument is not written in the caller.
int g(int x = G + 1);
// CHECK-LABEL: define {{.*}} @_Z6callerv(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap
int caller() {
  return g();
__overflow:
  return -1;
}

// Constructor initializers and default member initializers run before the
// body's handler is in scope. The base object constructor, emitted last, is
// checked at the end of the file.
struct D { D(); ~D(); };
struct S {
  int __ob_trap x;
  D d;
  int y = G + 2;
  S(int __ob_trap a) : x(a + 1) {
    return;
  __overflow:
    x = 0;
  }
};
S make(int a) { return S(a); }

// A branch out of a guarded initialization would leave the guard held.
// CHECK-LABEL: define {{.*}} @_Z12local_staticU8ObtTrap_i(
// CHECK: @__cxa_guard_acquire
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap
int local_static(int __ob_trap a) {
  static int s = a + 1;
  return s;
__overflow:
  return -1;
}

// A branch out of a new-expression's initializer or a throw-expression's
// operand would leak the allocation.
struct T { T(int); ~T(); };
// CHECK-LABEL: define {{.*}} @_Z8new_initU8ObtTrap_i(
// CHECK: @_Znwm
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap
T *new_init(int __ob_trap a) {
  return new T(a + 1);
__overflow:
  return nullptr;
}

// The array size is computed before the allocation and may branch.
// CHECK-LABEL: define {{.*}} @_Z8new_sizeU8ObtTrap_i(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %__overflow
int *new_size(int __ob_trap n) {
  return new int[n * 2];
__overflow:
  return nullptr;
}

// CHECK-LABEL: define {{.*}} @_Z9throw_endU8ObtTrap_i(
// CHECK: @__cxa_allocate_exception
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap
void throw_end(int __ob_trap a) {
  throw T(a + 1);
__overflow:
  return;
}

// CHECK-LABEL: define {{.*}} @_ZN1SC2EU8ObtTrap_i(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap
// CHECK: ret void
