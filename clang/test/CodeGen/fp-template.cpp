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


template <typename Ty>
Ty templ_02(Ty x, Ty y) {
  return x + y;
}

#pragma STDC FENV_ROUND FE_UPWARD

template <typename Ty>
Ty templ_03(Ty x, Ty y) {
  return x - y;
}

#pragma STDC FENV_ROUND FE_TONEAREST

float func_02(float x, float y) {
  return templ_02(x, y);
}

// CHECK-LABEL: define {{.*}} float @_Z8templ_02IfET_S0_S0_
// CHECK:       %add = fadd float %0, %1

float func_03(float x, float y) {
  return templ_03(x, y);
}

// CHECK-LABEL: define {{.*}} float @_Z8templ_03IfET_S0_S0_
// CHECK:       call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore")


// This pragma sets non-default rounding mode before delayed parsing occurs. It
// is used to check that the parsing uses FP options defined by command line
// options or by pragma before the template definition but not by this pragma.
#pragma STDC FENV_ROUND FE_TOWARDZERO


// CHECK: attributes #[[ATTR01]] = { {{.*}}strictfp
