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

typedef double vector4double __attribute__((__vector_size__(32)));
typedef float  vector4float  __attribute__((__vector_size__(16)));
vector4float func_21(vector4double x) {
  #pragma STDC FENV_ROUND FE_UPWARD
  return __builtin_convertvector(x, vector4float);
}
// CHECK-LABEL: @func_21
// STRICT: call <4 x float> @llvm.experimental.constrained.fptrunc.v4f32.v4f64(<4 x double> {{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")

typedef short vector8short __attribute__((__vector_size__(16)));
typedef double vector8double __attribute__((__vector_size__(64)));
vector8double func_24(vector8short x) {
  #pragma STDC FENV_ROUND FE_TOWARDZERO
  return __builtin_convertvector(x, vector8double);
}
// CHECK-LABEL: @func_24
// STRICT: call <8 x double> @llvm.experimental.constrained.sitofp.v8f64.v8i16(<8 x i16> {{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")

typedef unsigned int vector16uint __attribute__((__vector_size__(64)));
typedef double vector16double __attribute__((__vector_size__(128)));
vector16double func_25(vector16uint x) {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  return __builtin_convertvector(x, vector16double);
}
// CHECK-LABEL: @func_25
// STRICT: call <16 x double> @llvm.experimental.constrained.uitofp.v16f64.v16i32(<16 x i32> {{.*}}, metadata !"round.downward", metadata !"fpexcept.strict")

typedef float vector2float __attribute__((__vector_size__(8)));
typedef char vector2char __attribute__((__vector_size__(2)));
vector2char func_22(vector2float x) {
  #pragma float_control(except, off)
  return __builtin_convertvector(x, vector2char);
}
// CHECK-LABEL: @func_22
// STRICT: call <2 x i8> @llvm.experimental.constrained.fptosi.v2i8.v2f32(<2 x float> {{.*}}, metadata !"fpexcept.ignore")

typedef float vector3float __attribute__((__vector_size__(12)));
typedef unsigned long long vector3ulong __attribute__((__vector_size__(24)));
vector3ulong func_23(vector3float x) {
  #pragma float_control(except, off)
  return __builtin_convertvector(x, vector3ulong);
}
// CHECK-LABEL: @func_23
// STRICT: call <3 x i64> @llvm.experimental.constrained.fptoui.v3i64.v3f32(<3 x float> {{.*}}, metadata !"fpexcept.ignore")
