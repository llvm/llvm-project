// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fdelayed-template-parsing -o - %s | FileCheck %s

template <typename T>
T templ_01(T x, T y) {
#pragma STDC FENV_ACCESS ON
  return x + y;
}

float func_01(float x, float y) {
  return templ_01(x, y);
}

// CHECK-LABEL: define {{.*}} @_Z8templ_01IfET_S0_S0_
// CHECK-SAME:  (float noundef %{{.*}}, float noundef %{{.*}}) #[[ATTR01:[0-9]+]]{{.*}} {
// CHECK:       call float @llvm.experimental.constrained.fadd.f32

// CHECK: attributes #[[ATTR01]] = { {{.*}}strictfp
