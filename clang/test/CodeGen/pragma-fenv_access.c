// RUN: %clang_cc1 -fexperimental-strict-floating-point -ffp-exception-behavior=strict -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point -ffp-exception-behavior=strict -triple %itanium_abi_triple -emit-llvm %s -o - -fms-extensions -DMS | FileCheck %s

#ifdef MS
#pragma fenv_access (on)
#else
#pragma STDC FENV_ACCESS ON
#endif

float func_01(float x, float y) {
  return x + y;
}
// CHECK-LABEL: @func_01
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")


float func_02(float x, float y) {
  #pragma float_control(except, off)
  #pragma STDC FENV_ACCESS OFF
  return x + y;
}
// CHECK-LABEL: @func_02
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")


float func_03(float x, float y) {
  return x + y;
}
// CHECK-LABEL: @func_03
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")


#ifdef MS
#pragma fenv_access (off)
#else
#pragma STDC FENV_ACCESS OFF
#endif

float func_04(float x, float y) {
  #pragma float_control(except, off)
  return x + y;
}
// CHECK-LABEL: @func_04
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")


float func_05(float x, float y) {
  #pragma STDC FENV_ACCESS ON
  return x + y;
}
// CHECK-LABEL: @func_05
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")


float func_06(float x, float y) {
  #pragma float_control(except, off)
  return x + y;
}
// CHECK-LABEL: @func_06
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")


float func_07(float x, float y) {
  x -= y;
  if (x) {
    #pragma STDC FENV_ACCESS ON
    y *= 2;
  }
  return y + 4;
}
// CHECK-LABEL: @func_07
// CHECK: call float @llvm.experimental.constrained.fsub.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
