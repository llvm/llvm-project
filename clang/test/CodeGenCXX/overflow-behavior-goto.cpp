// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s

// The builtin is an lvalue when its operand is; checks in the operand still
// branch to the label.
// CHECK-LABEL: define {{.*}} @_Z6lvalueU8ObtTrap_i(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %l
// CHECK: store i32 3, ptr %a.addr
int lvalue(int __ob_trap a) {
  __builtin_overflow_goto(a += 1, l) = 3;
  return a;
l:
  return 0;
}

// Instantiation binds the instantiated label.
// CHECK-LABEL: define {{.*}} @_Z4instIU8ObtTrap_iEiT_(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %too_big
template <typename T> int inst(T a) {
  T r = __builtin_overflow_goto(a * 2, too_big);
  return r;
too_big:
  return -1;
}
template int inst(int __ob_trap);
