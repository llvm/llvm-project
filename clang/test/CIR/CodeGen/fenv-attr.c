// Verify that CIR attaches #cir.fenv to floating-point operations when
// constrained FP mode is in effect, and that lowering to LLVM IR matches
// classic codegen.

// --- -ffp-model=strict (cc1 equivalent) ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -ffp-contract=off -frounding-math -ffp-exception-behavior=strict \
// RUN:   %s -o %t-model.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-MODEL --input-file=%t-model.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -ffp-contract=off -frounding-math -ffp-exception-behavior=strict \
// RUN:   %s -o %t-model-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-MODEL --input-file=%t-model-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -ffp-contract=off -frounding-math -ffp-exception-behavior=strict \
// RUN:   %s -o %t-model.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-MODEL --input-file=%t-model.ll %s

// --- -ffp-exception-behavior=strict ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -ffp-exception-behavior=strict %s -o %t-strict.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-STRICT --input-file=%t-strict.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -ffp-exception-behavior=strict %s -o %t-strict-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT --input-file=%t-strict-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -ffp-exception-behavior=strict %s -o %t-strict.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-STRICT --input-file=%t-strict.ll %s

// --- -ffp-exception-behavior=maytrap ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -ffp-exception-behavior=maytrap %s -o %t-maytrap.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-MAYTRAP --input-file=%t-maytrap.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -ffp-exception-behavior=maytrap %s -o %t-maytrap-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-MAYTRAP --input-file=%t-maytrap-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -ffp-exception-behavior=maytrap %s -o %t-maytrap.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-MAYTRAP --input-file=%t-maytrap.ll %s

// --- -frounding-math (dynamic rounding, ignored exceptions) ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -frounding-math %s -o %t-rnd.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-RND --input-file=%t-rnd.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -frounding-math %s -o %t-rnd-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-RND --input-file=%t-rnd-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -frounding-math %s -o %t-rnd.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-RND --input-file=%t-rnd.ll %s

// --- pragma-only (no command-line constrained FP) ---
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -fexperimental-strict-floating-point %s -o %t-pragma.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-PRAGMA --input-file=%t-pragma.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -fexperimental-strict-floating-point %s -o %t-pragma-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-PRAGMA --input-file=%t-pragma-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -fexperimental-strict-floating-point %s -o %t-pragma.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-PRAGMA --input-file=%t-pragma.ll %s

float add(float x, float y) {
  return x + y;
}

// CIR-LABEL: @add
// CIR-MODEL: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.fadd {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.fadd {{.*}} : !cir.float loc
// LLVM-LABEL: @add
// LLVM-MODEL: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: fadd float

float sub(float x, float y) {
  return x - y;
}

// CIR-LABEL: @sub
// CIR-MODEL: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.fsub {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.fsub {{.*}} : !cir.float loc
// LLVM-LABEL: @sub
// LLVM-MODEL: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: fsub float

float mul(float x, float y) {
  return x * y;
}

// CIR-LABEL: @mul
// CIR-MODEL: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.fmul {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.fmul {{.*}} : !cir.float loc
// LLVM-LABEL: @mul
// LLVM-MODEL: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: fmul float

float div(float x, float y) {
  return x / y;
}

// CIR-LABEL: @div
// CIR-MODEL: cir.fdiv {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.fdiv {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.fdiv {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.fdiv {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.fdiv {{.*}} : !cir.float loc
// LLVM-LABEL: @div
// LLVM-MODEL: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: fdiv float

float use_sqrt(float x) {
  return __builtin_sqrtf(x);
}

// CIR-LABEL: @use_sqrt
// CIR-MODEL: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.sqrt {{.*}} : !cir.float loc
// LLVM-LABEL: @use_sqrt
// LLVM-MODEL: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.sqrt.f32

float use_cos(float x) {
  return __builtin_cosf(x);
}

// CIR-LABEL: @use_cos
// CIR-MODEL: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.cos {{.*}} : !cir.float loc
// LLVM-LABEL: @use_cos
// LLVM-MODEL: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.cos.f32

float use_sin(float x) {
  return __builtin_sinf(x);
}

// CIR-LABEL: @use_sin
// CIR-MODEL: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.sin {{.*}} : !cir.float loc
// LLVM-LABEL: @use_sin
// LLVM-MODEL: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.sin.f32

float use_pow(float x, float y) {
  return __builtin_powf(x, y);
}

// CIR-LABEL: @use_pow
// CIR-MODEL: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.pow {{.*}} : !cir.float loc
// LLVM-LABEL: @use_pow
// LLVM-MODEL: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.pow.f32

float sqrtf(float);
float cosf(float);
float sinf(float);
float powf(float, float);

float lib_sqrt(float x) {
  return sqrtf(x);
}

// CIR-LABEL: @lib_sqrt
// CIR-MODEL: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.sqrt {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.sqrt {{.*}} : !cir.float loc
// LLVM-LABEL: @lib_sqrt
// LLVM-MODEL: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.sqrt.f32

float lib_cos(float x) {
  return cosf(x);
}

// CIR-LABEL: @lib_cos
// CIR-MODEL: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.cos {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.cos {{.*}} : !cir.float loc
// LLVM-LABEL: @lib_cos
// LLVM-MODEL: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.cos.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.cos.f32

float lib_sin(float x) {
  return sinf(x);
}

// CIR-LABEL: @lib_sin
// CIR-MODEL: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.sin {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.sin {{.*}} : !cir.float loc
// LLVM-LABEL: @lib_sin
// LLVM-MODEL: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.sin.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.sin.f32

float lib_pow(float x, float y) {
  return powf(x, y);
}

// CIR-LABEL: @lib_pow
// CIR-MODEL: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-STRICT: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-MAYTRAP: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = false>}
// CIR-RND: cir.pow {{.*}} {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// CIR-PRAGMA: cir.pow {{.*}} : !cir.float loc
// LLVM-LABEL: @lib_pow
// LLVM-MODEL: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-MAYTRAP: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// LLVM-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// LLVM-PRAGMA: call float @llvm.pow.f32
