// Verify #pragma STDC FENV_ACCESS / float_control / clang fp exception and
// rounding pragmas produce the expected constrained FP behavior in CIR and in
// LLVM IR (both via -fclangir and classic codegen).

// --- STRICT: -ffp-exception-behavior=strict ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -ffp-exception-behavior=strict \
// RUN:   -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-STRICT --input-file=%t-strict.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -ffp-exception-behavior=strict \
// RUN:   -emit-llvm %s -o %t-strict-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT \
// RUN:   --input-file=%t-strict-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -fexperimental-strict-floating-point -ffp-exception-behavior=strict \
// RUN:   -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT \
// RUN:   --input-file=%t-strict.ll %s

// --- STRICT-RND: -frounding-math -ffp-exception-behavior=strict ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -ffp-exception-behavior=strict -emit-cir %s -o %t-strict-rnd.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-STRICT-RND \
// RUN:   --input-file=%t-strict-rnd.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-rnd-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT-RND \
// RUN:   --input-file=%t-strict-rnd-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-rnd.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT-RND \
// RUN:   --input-file=%t-strict-rnd.ll %s

// --- STRICT with MS fenv_access spelling ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -ffp-exception-behavior=strict \
// RUN:   -fms-extensions -DMS -emit-llvm %s -o %t-strict-ms-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT \
// RUN:   --input-file=%t-strict-ms-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -fexperimental-strict-floating-point -ffp-exception-behavior=strict \
// RUN:   -fms-extensions -DMS -emit-llvm %s -o %t-strict-ms.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT \
// RUN:   --input-file=%t-strict-ms.ll %s

// --- STRICT-RND with MS fenv_access spelling ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -ffp-exception-behavior=strict -fms-extensions -DMS \
// RUN:   -emit-llvm %s -o %t-strict-rnd-ms-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT-RND \
// RUN:   --input-file=%t-strict-rnd-ms-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -ffp-exception-behavior=strict -fms-extensions -DMS \
// RUN:   -emit-llvm %s -o %t-strict-rnd-ms.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT-RND \
// RUN:   --input-file=%t-strict-rnd-ms.ll %s

// --- DEFAULT: pragma-only constrained FP ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -emit-cir %s -o %t-default.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-DEFAULT --input-file=%t-default.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -emit-llvm %s -o %t-default-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-DEFAULT \
// RUN:   --input-file=%t-default-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -fexperimental-strict-floating-point -emit-llvm %s -o %t-default.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-DEFAULT \
// RUN:   --input-file=%t-default.ll %s

// --- DEFAULT-RND: -frounding-math, pragma-only exceptions ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -emit-cir %s -o %t-default-rnd.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-DEFAULT-RND \
// RUN:   --input-file=%t-default-rnd.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -emit-llvm %s -o %t-default-rnd-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-DEFAULT-RND \
// RUN:   --input-file=%t-default-rnd-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -fexperimental-strict-floating-point -frounding-math \
// RUN:   -emit-llvm %s -o %t-default-rnd.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-DEFAULT-RND \
// RUN:   --input-file=%t-default-rnd.ll %s

float test_cmdline_defaults(float x, float y) {
  return x + y;
}

// CIR-LABEL: @test_cmdline_defaults
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-LABEL: @test_cmdline_defaults
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")

// When #pragma float_control(except, on) is enabled in a nested scope,
// operations outside the scope still carry an fenv attribute, but with
// ignore-style exception settings derived from the ambient FP options.
// Placed before any global-scope FENV_ACCESS pragma so -frounding-math is
// not cleared by FENV_ACCESS OFF (setRoundingMathOverride(false)).
float test_scoped_float_control(float x, float y) {
  float a = x + y;
  {
#pragma float_control(except, on)
    float b = x * y;
    a = a + b;
  }
  return a - y;
}

// CIR-LABEL: @test_scoped_float_control
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT-RND: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-LABEL: @test_scoped_float_control
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")

#ifdef MS
#pragma fenv_access (on)
#else
#pragma STDC FENV_ACCESS ON
#endif

float test_global_fenv_access_on(float x, float y) {
  return x + y;
}

// CIR-LABEL: @test_global_fenv_access_on
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_global_fenv_access_on
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

float test_except_off_fenv_access_off(float x, float y) {
#pragma float_control(except, off)
#pragma STDC FENV_ACCESS OFF
  return x + y;
}

// CIR-LABEL: @test_except_off_fenv_access_off
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-LABEL: @test_except_off_fenv_access_off
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")

float test_fenv_access_on_after_local_off(float x, float y) {
  return x + y;
}

// CIR-LABEL: @test_fenv_access_on_after_local_off
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_fenv_access_on_after_local_off
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

#ifdef MS
#pragma fenv_access (off)
#else
#pragma STDC FENV_ACCESS OFF
#endif

float test_float_control_except_off(float x, float y) {
#pragma float_control(except, off)
  return x + y;
}

// CIR-LABEL: @test_float_control_except_off
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-LABEL: @test_float_control_except_off
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float

float test_float_control_except_on(float x, float y) {
#pragma float_control(except, on)
  return x + y;
}

// CIR-LABEL: @test_float_control_except_on
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_float_control_except_on
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

float test_local_fenv_access_on(float x, float y) {
#pragma STDC FENV_ACCESS ON
  return x + y;
}

// CIR-LABEL: @test_local_fenv_access_on
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_local_fenv_access_on
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

float test_float_control_except_off_after_on(float x, float y) {
#pragma float_control(except, off)
  return x + y;
}

// CIR-LABEL: @test_float_control_except_off_after_on
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-LABEL: @test_float_control_except_off_after_on
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float

float test_scoped_fenv_access_on(float x, float y) {
  x -= y;
  if (x) {
#pragma STDC FENV_ACCESS ON
    y *= 2.0F;
  }
  return y + 4.0F;
}

// CIR-LABEL: @test_scoped_fenv_access_on
// CIR-STRICT: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-LABEL: @test_scoped_fenv_access_on
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")

float test_fenv_round_upward(float x, float y) {
#pragma STDC FENV_ROUND FE_UPWARD
#pragma STDC FENV_ACCESS ON
  return x + y;
}

// CIR-LABEL: @test_fenv_round_upward
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_fenv_round_upward
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")

float test_fenv_round_tonearest(float x, float y) {
#pragma STDC FENV_ROUND FE_TONEAREST
#pragma STDC FENV_ACCESS ON
  return x + y;
}

// CIR-LABEL: @test_fenv_round_tonearest
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_fenv_round_tonearest
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

float test_fenv_round_tonearest_except_ignore(float x, float y) {
#pragma STDC FENV_ROUND FE_TONEAREST
#pragma clang fp exceptions(ignore)
#pragma STDC FENV_ACCESS ON
  return x + y;
}

// CIR-LABEL: @test_fenv_round_tonearest_except_ignore
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-LABEL: @test_fenv_round_tonearest_except_ignore
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")

float test_fenv_round_tonearest_except_ignore_fenv_off(float x, float y) {
#pragma STDC FENV_ROUND FE_TONEAREST
#pragma clang fp exceptions(ignore)
#pragma STDC FENV_ACCESS OFF
  return x + y;
}

// CIR-LABEL: @test_fenv_round_tonearest_except_ignore_fenv_off
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-LABEL: @test_fenv_round_tonearest_except_ignore_fenv_off
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float

float test_fp_exceptions_maytrap(float x, float y) {
#pragma clang fp exceptions(maytrap)
#pragma STDC FENV_ACCESS ON
  return x + y;
}

// CIR-LABEL: @test_fp_exceptions_maytrap
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM-LABEL: @test_fp_exceptions_maytrap
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")

float test_fp_exceptions_maytrap_round_upward(float x, float y) {
#pragma clang fp exceptions(maytrap)
#pragma STDC FENV_ROUND FE_UPWARD
#pragma STDC FENV_ACCESS ON
  return x + y;
}

// CIR-LABEL: @test_fp_exceptions_maytrap_round_upward
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM-LABEL: @test_fp_exceptions_maytrap_round_upward
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")

float test_nested_fenv_access_off(float x, float y, float z) {
#pragma STDC FENV_ACCESS ON
  float res = x * y;
  {
#pragma STDC FENV_ACCESS OFF
    return res + z;
  }
}

// CIR-LABEL: @test_nested_fenv_access_off
// CIR-STRICT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-LABEL: @test_nested_fenv_access_off
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")

float test_fenv_round_towardzero_nested_fenv_off(float x, float y, float z) {
#pragma STDC FENV_ROUND FE_TOWARDZERO
#pragma STDC FENV_ACCESS ON
  float res = x * y;
  {
#pragma STDC FENV_ACCESS OFF
    return res + z;
  }
}

// CIR-LABEL: @test_fenv_round_towardzero_nested_fenv_off
// CIR-STRICT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = masked, strict_except = false>}
// LLVM-LABEL: @test_fenv_round_towardzero_nested_fenv_off
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")

float test_nested_fenv_round_then_fenv_access(float x, float y) {
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

// CIR-LABEL: @test_nested_fenv_round_then_fenv_access
// CIR-STRICT: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT-RND: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_nested_fenv_round_then_fenv_access
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

float test_fenv_round_dynamic_with_fenv_access(float x, float y) {
#pragma STDC FENV_ROUND FE_DYNAMIC
#pragma STDC FENV_ACCESS ON
  return x + y;
}

// CIR-LABEL: @test_fenv_round_dynamic_with_fenv_access
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_fenv_round_dynamic_with_fenv_access
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

float test_fenv_round_dynamic_without_fenv_access(float x, float y) {
#pragma STDC FENV_ROUND FE_DYNAMIC
  return x + y;
}

// CIR-LABEL: @test_fenv_round_dynamic_without_fenv_access
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-LABEL: @test_fenv_round_dynamic_without_fenv_access
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float

#pragma STDC FENV_ACCESS ON
float test_file_scope_fenv_access_on(float x, float y) {
  return x + y;
}

// CIR-LABEL: @test_file_scope_fenv_access_on
// CIR: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-LABEL: @test_file_scope_fenv_access_on
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

#pragma STDC FENV_ACCESS OFF
float test_file_scope_fenv_access_off(float x, float y) {
  return x + y;
}

// CIR-LABEL: @test_file_scope_fenv_access_off
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-LABEL: @test_file_scope_fenv_access_off
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float
