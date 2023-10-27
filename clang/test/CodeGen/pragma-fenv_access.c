// RUN: %clang_cc1 -fexperimental-strict-floating-point -ffp-exception-behavior=strict -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,STRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point -frounding-math -ffp-exception-behavior=strict -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,STRICT-RND %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point -ffp-exception-behavior=strict -triple %itanium_abi_triple -emit-llvm %s -o - -fms-extensions -DMS | FileCheck --check-prefixes=CHECK,STRICT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point -frounding-math -ffp-exception-behavior=strict -triple %itanium_abi_triple -emit-llvm %s -o - -fms-extensions -DMS | FileCheck --check-prefixes=CHECK,STRICT-RND %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point -frounding-math -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,DEFAULT-RND %s

float func_00(float x, float y) {
  return x + y;
}
// CHECK-LABEL: @func_00
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT-RND: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// DEFAULT: fadd float


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
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")


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
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// DEFAULT: fadd float


float func_04a(float x, float y) {
  #pragma float_control(except, on)
  return x + y;
}
// CHECK-LABEL: @func_04a
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")


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
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// DEFAULT: fadd float


float func_07(float x, float y) {
  x -= y;
  if (x) {
    #pragma STDC FENV_ACCESS ON
    y *= 2.0F;
  }
  return y + 4.0F;
}
// CHECK-LABEL: @func_07
// STRICT: call float @llvm.experimental.constrained.fsub.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fsub.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// DEFAULT: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")


float func_08(float x, float y) {
  #pragma STDC FENV_ROUND FE_UPWARD
  #pragma STDC FENV_ACCESS ON
  return x + y;
}
// CHECK-LABEL: @func_08
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")


float func_09(float x, float y) {
  #pragma STDC FENV_ROUND FE_TONEAREST
  #pragma STDC FENV_ACCESS ON
  return x + y;
}
// CHECK-LABEL: @func_09
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")


float func_10(float x, float y) {
  #pragma STDC FENV_ROUND FE_TONEAREST
  #pragma clang fp exceptions(ignore)
  #pragma STDC FENV_ACCESS ON
  return x + y;
}
// CHECK-LABEL: @func_10
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")


float func_11(float x, float y) {
  #pragma STDC FENV_ROUND FE_TONEAREST
  #pragma clang fp exceptions(ignore)
  #pragma STDC FENV_ACCESS OFF
  return x + y;
}
// CHECK-LABEL: @func_11
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// DEFAULT: fadd float


float func_12(float x, float y) {
  #pragma clang fp exceptions(maytrap)
  #pragma STDC FENV_ACCESS ON
  return x + y;
}
// CHECK-LABEL: @func_12
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")


float func_13(float x, float y) {
  #pragma clang fp exceptions(maytrap)
  #pragma STDC FENV_ROUND FE_UPWARD
  #pragma STDC FENV_ACCESS ON
  return x + y;
}
// CHECK-LABEL: @func_13
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")


float func_14(float x, float y, float z) {
  #pragma STDC FENV_ACCESS ON
  float res = x * y;
  {
    #pragma STDC FENV_ACCESS OFF
    return res + z;
  }
}
// CHECK-LABEL: @func_14
// STRICT:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// STRICT:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")


float func_15(float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_TOWARDZERO
  #pragma STDC FENV_ACCESS ON
  float res = x * y;
  {
    #pragma STDC FENV_ACCESS OFF
    return res + z;
  }
}
// CHECK-LABEL: @func_15
// STRICT:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// STRICT:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")


float func_16(float x, float y) {
  x -= y;
  {
    #pragma STDC FENV_ROUND FE_TONEAREST
    #pragma STDC FENV_ACCESS ON
    y *= 2.0F;
  }
  {
    #pragma STDC FENV_ACCESS ON
    return y + 4.0F;
  }
}
// CHECK-LABEL: @func_16
// STRICT: call float @llvm.experimental.constrained.fsub.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fsub.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// DEFAULT: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// DEFAULT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")


float func_17(float x, float y) {
  #pragma STDC FENV_ROUND FE_DYNAMIC
  #pragma STDC FENV_ACCESS ON
  return x + y;
}
// CHECK-LABEL: @func_17
// CHECK: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")


float func_18(float x, float y) {
  #pragma STDC FENV_ROUND FE_DYNAMIC
  return x + y;
}
// CHECK-LABEL: @func_18
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// DEFAULT: fadd float

#pragma STDC FENV_ACCESS ON
float func_19(float x, float y) {
  return x + y;
}
// CHECK-LABEL: @func_19
// STRICT:  call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

#pragma STDC FENV_ACCESS OFF
float func_20(float x, float y) {
  return x + y;
}
// CHECK-LABEL: @func_20
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// DEFAULT: fadd float
