// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s

// A lambda is its own function: it cannot reach the enclosing function's
// handler, but may define its own.
int f(int __ob_trap a) {
  auto no_handler = [=]() -> int { return a + 1; };
  auto own_handler = [=]() -> int {
    return a * 2;
  __overflow:
    return -9;
  };
  return no_handler() + own_handler();
__overflow:
  return -1;
}

// CHECK-LABEL: define {{.*}} @_Z1fU8ObtTrap_i(
// CHECK-NOT: br label %__overflow
// CHECK: ret i32

// CHECK-LABEL: define internal {{.*}} @"_ZZ1fU8ObtTrap_iENK3$_0clEv"(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: call void @llvm.ubsantrap

// CHECK-LABEL: define internal {{.*}} @"_ZZ1fU8ObtTrap_iENK3$_1clEv"(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %__overflow

template <class T> T twice(T x) {
  return x + x;
__overflow:
  return T(0);
}

int g(int __ob_trap a) { return twice(a); }

// CHECK-LABEL: define linkonce_odr {{.*}} @_Z5twiceIU8ObtTrap_iET_S1_(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %__overflow
