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
// RUN: FileCheck --check-prefixes=CIR,CIR-STRICT-RND --input-file=%t-strict-rnd.cir %s
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
// RUN: FileCheck --check-prefixes=CIR,CIR-DEFAULT-RND --input-file=%t-default-rnd.cir %s
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

void test_cmdline_defaults(float x, float y, float z, int i, double d) {
// CIR-LABEL: cir.func {{.*}}@test_cmdline_defaults(
// CIR-STRICT-SAME: strictfp
// CIR-STRICT-RND-SAME: strictfp
// CIR-DEFAULT-RND-SAME: strictfp
// CIR-DEFAULT-NOT: strictfp
// LLVM-LABEL: define {{.*}}@test_cmdline_defaults(
// LLVM-STRICT-SAME: #[[$STRICT_ATTR:[0-9]+]]
// LLVM-STRICT-RND-SAME: #[[$STRICT_ATTR:[0-9]+]]
// LLVM-DEFAULT-RND-SAME: #[[$STRICT_ATTR:[0-9]+]]
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sf = x - y;
// CIR-STRICT: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fsub {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fsub float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sf = x * y;
// CIR-STRICT: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmul {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fmul float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sf = x / y;
// CIR-STRICT: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fdiv {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fdiv float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR-STRICT: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cmp lt {{.*}} : !cir.float
// CIR-DEFAULT-RND: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fcmp olt float
// LLVM-DEFAULT-RND: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
  sf = (float)i;
// CIR-STRICT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// CIR-DEFAULT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: sitofp i32 {{.*}} to float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  si = (int)x;
// CIR-STRICT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// CIR-DEFAULT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: fptosi float {{.*}} to i32
// LLVM-DEFAULT-RND: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
  sd = (double)x;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.float -> !cir.double
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: fpext float {{.*}} to double
// LLVM-DEFAULT-RND: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
  sf = (float)d;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.double -> !cir.float
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fptrunc double {{.*}} to float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sf = __builtin_sqrtf(x);
// CIR-STRICT: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.sqrt {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.sqrt.f32
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sf = __builtin_powf(x, y);
// CIR-STRICT: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.pow {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.pow.f32
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sf = __builtin_fmaf(x, y, z);
// CIR-STRICT: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fma {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.fma.f32
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  sl = __builtin_lroundf(x);
// CIR-STRICT: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.lround {{.*}} : !cir.float -> !s64i
// CIR-DEFAULT-RND: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: call i64 @llvm.lround.i64.f32
// LLVM-DEFAULT-RND: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
  sf = __builtin_fmodf(x, y);
// CIR-STRICT: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmod {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: frem float
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
}

#pragma STDC FENV_ACCESS ON

void test_global_fenv_access_on(float x, float y, float z, int i, double d) {
// CIR-LABEL: cir.func {{.*}}@test_global_fenv_access_on(
// LLVM-LABEL: define {{.*}}@test_global_fenv_access_on(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
}

void test_except_off_fenv_access_off(float x, float y, float z, int i, double d) {
#pragma float_control(except, off)
#pragma STDC FENV_ACCESS OFF
// CIR-LABEL: cir.func {{.*}}@test_except_off_fenv_access_off(
// LLVM-LABEL: define {{.*}}@test_except_off_fenv_access_off(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
}

void test_fenv_access_on_after_local_off(float x, float y, float z, int i, double d) {
// CIR-LABEL: cir.func {{.*}}@test_fenv_access_on_after_local_off(
// LLVM-LABEL: define {{.*}}@test_fenv_access_on_after_local_off(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
}

#pragma STDC FENV_ACCESS OFF

void test_float_control_except_off(float x, float y, float z, int i, double d) {
#pragma float_control(except, off)
// CIR-LABEL: cir.func {{.*}}@test_float_control_except_off(
// LLVM-LABEL: define {{.*}}@test_float_control_except_off(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float
  sf = x - y;
// CIR-STRICT: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fsub {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fsub {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fsub float
// LLVM-DEFAULT-RND: fsub float
  sf = x * y;
// CIR-STRICT: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmul {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmul {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fmul float
// LLVM-DEFAULT-RND: fmul float
  sf = x / y;
// CIR-STRICT: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fdiv {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fdiv {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fdiv float
// LLVM-DEFAULT-RND: fdiv float
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR-STRICT: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cmp lt {{.*}} : !cir.float
// CIR-DEFAULT-RND: cir.cmp lt {{.*}} : !cir.float
// LLVM-STRICT: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fcmp olt float
// LLVM-DEFAULT-RND: fcmp olt float
  sf = (float)i;
// CIR-STRICT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// CIR-DEFAULT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: sitofp i32 {{.*}} to float
// LLVM-DEFAULT-RND: sitofp i32 {{.*}} to float
  si = (int)x;
// CIR-STRICT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// CIR-DEFAULT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// LLVM-STRICT: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fptosi float {{.*}} to i32
// LLVM-DEFAULT-RND: fptosi float {{.*}} to i32
  sd = (double)x;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.float -> !cir.double
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double
// LLVM-STRICT: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fpext float {{.*}} to double
// LLVM-DEFAULT-RND: fpext float {{.*}} to double
  sf = (float)d;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.double -> !cir.float
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fptrunc double {{.*}} to float
// LLVM-DEFAULT-RND: fptrunc double {{.*}} to float
  sf = __builtin_sqrtf(x);
// CIR-STRICT: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.sqrt {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.sqrt {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.sqrt.f32
// LLVM-DEFAULT-RND: call float @llvm.sqrt.f32
  sf = __builtin_powf(x, y);
// CIR-STRICT: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.pow {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.pow {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.pow.f32
// LLVM-DEFAULT-RND: call float @llvm.pow.f32
  sf = __builtin_fmaf(x, y, z);
// CIR-STRICT: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fma {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fma {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.fma.f32
// LLVM-DEFAULT-RND: call float @llvm.fma.f32
  sl = __builtin_lroundf(x);
// CIR-STRICT: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.lround {{.*}} : !cir.float -> !s64i
// CIR-DEFAULT-RND: cir.lround {{.*}} : !cir.float -> !s64i
// LLVM-STRICT: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call i64 @llvm.lround.i64.f32
// LLVM-DEFAULT-RND: call i64 @llvm.lround.i64.f32
  sf = __builtin_fmodf(x, y);
// CIR-STRICT: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmod {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmod {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: frem float
// LLVM-DEFAULT-RND: frem float
}

void test_float_control_except_on(float x, float y, float z, int i, double d) {
#pragma float_control(except, on)
// CIR-LABEL: cir.func {{.*}}@test_float_control_except_on(
// LLVM-LABEL: define {{.*}}@test_float_control_except_on(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
}

void test_local_fenv_access_on(float x, float y, float z, int i, double d) {
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_local_fenv_access_on(
// LLVM-LABEL: define {{.*}}@test_local_fenv_access_on(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
}

void test_float_control_except_off_after_on(float x, float y, float z, int i, double d) {
#pragma float_control(except, off)
// CIR-LABEL: cir.func {{.*}}@test_float_control_except_off_after_on(
// LLVM-LABEL: define {{.*}}@test_float_control_except_off_after_on(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float
  sf = x - y;
// CIR-STRICT: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fsub {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fsub {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fsub float
// LLVM-DEFAULT-RND: fsub float
  sf = x * y;
// CIR-STRICT: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmul {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmul {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fmul float
// LLVM-DEFAULT-RND: fmul float
  sf = x / y;
// CIR-STRICT: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fdiv {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fdiv {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fdiv float
// LLVM-DEFAULT-RND: fdiv float
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR-STRICT: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cmp lt {{.*}} : !cir.float
// CIR-DEFAULT-RND: cir.cmp lt {{.*}} : !cir.float
// LLVM-STRICT: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fcmp olt float
// LLVM-DEFAULT-RND: fcmp olt float
  sf = (float)i;
// CIR-STRICT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// CIR-DEFAULT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: sitofp i32 {{.*}} to float
// LLVM-DEFAULT-RND: sitofp i32 {{.*}} to float
  si = (int)x;
// CIR-STRICT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// CIR-DEFAULT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// LLVM-STRICT: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fptosi float {{.*}} to i32
// LLVM-DEFAULT-RND: fptosi float {{.*}} to i32
  sd = (double)x;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.float -> !cir.double
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double
// LLVM-STRICT: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fpext float {{.*}} to double
// LLVM-DEFAULT-RND: fpext float {{.*}} to double
  sf = (float)d;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.double -> !cir.float
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fptrunc double {{.*}} to float
// LLVM-DEFAULT-RND: fptrunc double {{.*}} to float
  sf = __builtin_sqrtf(x);
// CIR-STRICT: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.sqrt {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.sqrt {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.sqrt.f32
// LLVM-DEFAULT-RND: call float @llvm.sqrt.f32
  sf = __builtin_powf(x, y);
// CIR-STRICT: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.pow {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.pow {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.pow.f32
// LLVM-DEFAULT-RND: call float @llvm.pow.f32
  sf = __builtin_fmaf(x, y, z);
// CIR-STRICT: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fma {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fma {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.fma.f32
// LLVM-DEFAULT-RND: call float @llvm.fma.f32
  sl = __builtin_lroundf(x);
// CIR-STRICT: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.lround {{.*}} : !cir.float -> !s64i
// CIR-DEFAULT-RND: cir.lround {{.*}} : !cir.float -> !s64i
// LLVM-STRICT: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call i64 @llvm.lround.i64.f32
// LLVM-DEFAULT-RND: call i64 @llvm.lround.i64.f32
  sf = __builtin_fmodf(x, y);
// CIR-STRICT: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmod {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmod {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: frem float
// LLVM-DEFAULT-RND: frem float
}

void test_scoped_fenv_access_on(float x, float y) {
// CIR-LABEL: cir.func {{.*}}@test_scoped_fenv_access_on(
// LLVM-LABEL: define {{.*}}@test_scoped_fenv_access_on(
  float sf;
  if (x) {
#pragma STDC FENV_ACCESS ON
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  }
}

void test_fenv_round_upward(float x, float y, float z, int i, double d) {
#pragma STDC FENV_ROUND FE_UPWARD
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_fenv_round_upward(
// LLVM-LABEL: define {{.*}}@test_fenv_round_upward(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
}

void test_fenv_round_tonearest(float x, float y, float z, int i, double d) {
#pragma STDC FENV_ROUND FE_TONEAREST
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_fenv_round_tonearest(
// LLVM-LABEL: define {{.*}}@test_fenv_round_tonearest(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
}

void test_fenv_round_tonearest_except_ignore(float x, float y, float z, int i, double d) {
#pragma STDC FENV_ROUND FE_TONEAREST
#pragma clang fp exceptions(ignore)
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_fenv_round_tonearest_except_ignore(
// LLVM-LABEL: define {{.*}}@test_fenv_round_tonearest_except_ignore(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
}

void test_fenv_round_tonearest_except_ignore_fenv_off(float x, float y, float z, int i, double d) {
#pragma STDC FENV_ROUND FE_TONEAREST
#pragma clang fp exceptions(ignore)
#pragma STDC FENV_ACCESS OFF
// CIR-LABEL: cir.func {{.*}}@test_fenv_round_tonearest_except_ignore_fenv_off(
// LLVM-LABEL: define {{.*}}@test_fenv_round_tonearest_except_ignore_fenv_off(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float
  sf = x - y;
// CIR-STRICT: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fsub {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fsub {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fsub float
// LLVM-DEFAULT-RND: fsub float
  sf = x * y;
// CIR-STRICT: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmul {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmul {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fmul float
// LLVM-DEFAULT-RND: fmul float
  sf = x / y;
// CIR-STRICT: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fdiv {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fdiv {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fdiv float
// LLVM-DEFAULT-RND: fdiv float
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR-STRICT: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cmp lt {{.*}} : !cir.float
// CIR-DEFAULT-RND: cir.cmp lt {{.*}} : !cir.float
// LLVM-STRICT: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fcmp olt float
// LLVM-DEFAULT-RND: fcmp olt float
  sf = (float)i;
// CIR-STRICT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// CIR-DEFAULT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: sitofp i32 {{.*}} to float
// LLVM-DEFAULT-RND: sitofp i32 {{.*}} to float
  si = (int)x;
// CIR-STRICT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// CIR-DEFAULT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// LLVM-STRICT: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fptosi float {{.*}} to i32
// LLVM-DEFAULT-RND: fptosi float {{.*}} to i32
  sd = (double)x;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.float -> !cir.double
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double
// LLVM-STRICT: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fpext float {{.*}} to double
// LLVM-DEFAULT-RND: fpext float {{.*}} to double
  sf = (float)d;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.double -> !cir.float
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: fptrunc double {{.*}} to float
// LLVM-DEFAULT-RND: fptrunc double {{.*}} to float
  sf = __builtin_sqrtf(x);
// CIR-STRICT: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.sqrt {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.sqrt {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.sqrt.f32
// LLVM-DEFAULT-RND: call float @llvm.sqrt.f32
  sf = __builtin_powf(x, y);
// CIR-STRICT: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.pow {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.pow {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.pow.f32
// LLVM-DEFAULT-RND: call float @llvm.pow.f32
  sf = __builtin_fmaf(x, y, z);
// CIR-STRICT: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fma {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fma {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call float @llvm.fma.f32
// LLVM-DEFAULT-RND: call float @llvm.fma.f32
  sl = __builtin_lroundf(x);
// CIR-STRICT: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.lround {{.*}} : !cir.float -> !s64i
// CIR-DEFAULT-RND: cir.lround {{.*}} : !cir.float -> !s64i
// LLVM-STRICT: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.ignore")
// LLVM-DEFAULT: call i64 @llvm.lround.i64.f32
// LLVM-DEFAULT-RND: call i64 @llvm.lround.i64.f32
  sf = __builtin_fmodf(x, y);
// CIR-STRICT: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-STRICT-RND: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT: cir.fmod {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmod {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT: frem float
// LLVM-DEFAULT-RND: frem float
}

void test_fp_exceptions_maytrap(float x, float y, float z, int i, double d) {
#pragma clang fp exceptions(maytrap)
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_fp_exceptions_maytrap(
// LLVM-LABEL: define {{.*}}@test_fp_exceptions_maytrap(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.maytrap")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.maytrap")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.maytrap")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.maytrap")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
}

void test_fp_exceptions_maytrap_round_upward(float x, float y, float z, int i, double d) {
#pragma clang fp exceptions(maytrap)
#pragma STDC FENV_ROUND FE_UPWARD
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_fp_exceptions_maytrap_round_upward(
// LLVM-LABEL: define {{.*}}@test_fp_exceptions_maytrap_round_upward(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.maytrap")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.maytrap")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.maytrap")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.maytrap")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upward, except_mode = unknown, strict_except = false>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.maytrap")
}

void test_nested_fenv_access_off(float x, float y) {
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_nested_fenv_access_off(
// LLVM-LABEL: define {{.*}}@test_nested_fenv_access_off(
  float sf;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  {
#pragma STDC FENV_ACCESS OFF
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  }
}

void test_fenv_round_towardzero_nested_fenv_off(float x, float y) {
#pragma STDC FENV_ROUND FE_TOWARDZERO
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_fenv_round_towardzero_nested_fenv_off(
// LLVM-LABEL: define {{.*}}@test_fenv_round_towardzero_nested_fenv_off(
  float sf;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
  {
#pragma STDC FENV_ACCESS OFF
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = upwardzero, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
  }
}

void test_nested_fenv_round_then_fenv_access(float x, float y) {
// CIR-LABEL: cir.func {{.*}}@test_nested_fenv_round_then_fenv_access(
// LLVM-LABEL: define {{.*}}@test_nested_fenv_round_then_fenv_access(
  float sf;
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  {
#pragma STDC FENV_ROUND FE_TONEAREST
#pragma STDC FENV_ACCESS ON
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  }
  {
#pragma STDC FENV_ACCESS ON
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  }
}

void test_fenv_round_dynamic_with_fenv_access(float x, float y, float z, int i, double d) {
#pragma STDC FENV_ROUND FE_DYNAMIC
#pragma STDC FENV_ACCESS ON
// CIR-LABEL: cir.func {{.*}}@test_fenv_round_dynamic_with_fenv_access(
// LLVM-LABEL: define {{.*}}@test_fenv_round_dynamic_with_fenv_access(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
}

void test_fenv_round_dynamic_without_fenv_access(float x, float y, float z, int i, double d) {
#pragma STDC FENV_ROUND FE_DYNAMIC
// CIR-LABEL: cir.func {{.*}}@test_fenv_round_dynamic_without_fenv_access(
// LLVM-LABEL: define {{.*}}@test_fenv_round_dynamic_without_fenv_access(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float
  sf = x - y;
// CIR-STRICT: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fsub {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fsub {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fsub float
// LLVM-DEFAULT-RND: fsub float
  sf = x * y;
// CIR-STRICT: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmul {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmul {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fmul float
// LLVM-DEFAULT-RND: fmul float
  sf = x / y;
// CIR-STRICT: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fdiv {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fdiv {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fdiv float
// LLVM-DEFAULT-RND: fdiv float
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR-STRICT: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cmp lt {{.*}} : !cir.float
// CIR-DEFAULT-RND: cir.cmp lt {{.*}} : !cir.float
// LLVM-STRICT: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fcmp olt float
// LLVM-DEFAULT-RND: fcmp olt float
  sf = (float)i;
// CIR-STRICT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// CIR-DEFAULT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: sitofp i32 {{.*}} to float
// LLVM-DEFAULT-RND: sitofp i32 {{.*}} to float
  si = (int)x;
// CIR-STRICT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// CIR-DEFAULT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// LLVM-STRICT: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: fptosi float {{.*}} to i32
// LLVM-DEFAULT-RND: fptosi float {{.*}} to i32
  sd = (double)x;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.float -> !cir.double
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double
// LLVM-STRICT: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: fpext float {{.*}} to double
// LLVM-DEFAULT-RND: fpext float {{.*}} to double
  sf = (float)d;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.double -> !cir.float
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fptrunc double {{.*}} to float
// LLVM-DEFAULT-RND: fptrunc double {{.*}} to float
  sf = __builtin_sqrtf(x);
// CIR-STRICT: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.sqrt {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.sqrt {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.sqrt.f32
// LLVM-DEFAULT-RND: call float @llvm.sqrt.f32
  sf = __builtin_powf(x, y);
// CIR-STRICT: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.pow {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.pow {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.pow.f32
// LLVM-DEFAULT-RND: call float @llvm.pow.f32
  sf = __builtin_fmaf(x, y, z);
// CIR-STRICT: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fma {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fma {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.fma.f32
// LLVM-DEFAULT-RND: call float @llvm.fma.f32
  sl = __builtin_lroundf(x);
// CIR-STRICT: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.lround {{.*}} : !cir.float -> !s64i
// CIR-DEFAULT-RND: cir.lround {{.*}} : !cir.float -> !s64i
// LLVM-STRICT: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: call i64 @llvm.lround.i64.f32
// LLVM-DEFAULT-RND: call i64 @llvm.lround.i64.f32
  sf = __builtin_fmodf(x, y);
// CIR-STRICT: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmod {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmod {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: frem float
// LLVM-DEFAULT-RND: frem float
}

#pragma STDC FENV_ACCESS ON
void test_file_scope_fenv_access_on(float x, float y, float z, int i, double d) {
// CIR-LABEL: cir.func {{.*}}@test_file_scope_fenv_access_on(
// LLVM-LABEL: define {{.*}}@test_file_scope_fenv_access_on(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x - y;
// CIR: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x * y;
// CIR: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = x / y;
// CIR: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  sf = (float)i;
// CIR: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  si = (int)x;
// CIR: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
  sd = (double)x;
// CIR: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = (float)d;
// CIR: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_sqrtf(x);
// CIR: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_powf(x, y);
// CIR: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sf = __builtin_fmaf(x, y, z);
// CIR: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM-DEFAULT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  sl = __builtin_lroundf(x);
// CIR: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
  sf = __builtin_fmodf(x, y);
// CIR: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>}
// LLVM: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
}

#pragma STDC FENV_ACCESS OFF
void test_file_scope_fenv_access_off(float x, float y, float z, int i, double d) {
// CIR-LABEL: cir.func {{.*}}@test_file_scope_fenv_access_off(
// LLVM-LABEL: define {{.*}}@test_file_scope_fenv_access_off(
  float sf;
  double sd;
  int si;
  long sl;
  _Bool sb;
  sf = x + y;
// CIR-STRICT: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fadd {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fadd {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fadd {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fadd float
// LLVM-DEFAULT-RND: fadd float
  sf = x - y;
// CIR-STRICT: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fsub {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fsub {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fsub {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fsub float
// LLVM-DEFAULT-RND: fsub float
  sf = x * y;
// CIR-STRICT: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmul {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmul {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmul {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fmul float
// LLVM-DEFAULT-RND: fmul float
  sf = x / y;
// CIR-STRICT: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fdiv {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fdiv {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fdiv {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fdiv float
// LLVM-DEFAULT-RND: fdiv float
  sf = -x;
// CIR: cir.fneg {{.*}} : !cir.float loc
// LLVM: fneg float
  sb = x < y;
// CIR-STRICT: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cmp lt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cmp lt {{.*}} : !cir.float
// CIR-DEFAULT-RND: cir.cmp lt {{.*}} : !cir.float
// LLVM-STRICT: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i1 @llvm.experimental.constrained.fcmps.f32({{.*}}, metadata !"olt", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fcmp olt float
// LLVM-DEFAULT-RND: fcmp olt float
  sf = (float)i;
// CIR-STRICT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// CIR-DEFAULT-RND: cir.cast int_to_float {{.*}} : !s32i -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: sitofp i32 {{.*}} to float
// LLVM-DEFAULT-RND: sitofp i32 {{.*}} to float
  si = (int)x;
// CIR-STRICT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// CIR-DEFAULT-RND: cir.cast float_to_int {{.*}} : !cir.float -> !s32i
// LLVM-STRICT: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i32 @llvm.experimental.constrained.fptosi.i32.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: fptosi float {{.*}} to i32
// LLVM-DEFAULT-RND: fptosi float {{.*}} to i32
  sd = (double)x;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.float -> !cir.double
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.float -> !cir.double
// LLVM-STRICT: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call double @llvm.experimental.constrained.fpext.f64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: fpext float {{.*}} to double
// LLVM-DEFAULT-RND: fpext float {{.*}} to double
  sf = (float)d;
// CIR-STRICT: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.cast floating {{.*}} : !cir.double -> !cir.float
// CIR-DEFAULT-RND: cir.cast floating {{.*}} : !cir.double -> !cir.float
// LLVM-STRICT: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fptrunc.f32.f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: fptrunc double {{.*}} to float
// LLVM-DEFAULT-RND: fptrunc double {{.*}} to float
  sf = __builtin_sqrtf(x);
// CIR-STRICT: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.sqrt {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.sqrt {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.sqrt {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.sqrt.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.sqrt.f32
// LLVM-DEFAULT-RND: call float @llvm.sqrt.f32
  sf = __builtin_powf(x, y);
// CIR-STRICT: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.pow {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.pow {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.pow {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.pow.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.pow.f32
// LLVM-DEFAULT-RND: call float @llvm.pow.f32
  sf = __builtin_fmaf(x, y, z);
// CIR-STRICT: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fma {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fma {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fma {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.fma.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: call float @llvm.fma.f32
// LLVM-DEFAULT-RND: call float @llvm.fma.f32
  sl = __builtin_lroundf(x);
// CIR-STRICT: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.lround {{.*}} : !cir.float -> !s64i {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.lround {{.*}} : !cir.float -> !s64i
// CIR-DEFAULT-RND: cir.lround {{.*}} : !cir.float -> !s64i
// LLVM-STRICT: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call i64 @llvm.experimental.constrained.lround.i64.f32({{.*}}, metadata !"fpexcept.strict")
// LLVM-DEFAULT: call i64 @llvm.lround.i64.f32
// LLVM-DEFAULT-RND: call i64 @llvm.lround.i64.f32
  sf = __builtin_fmodf(x, y);
// CIR-STRICT: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-STRICT-RND: cir.fmod {{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR-DEFAULT: cir.fmod {{.*}} : !cir.float loc
// CIR-DEFAULT-RND: cir.fmod {{.*}} : !cir.float loc
// LLVM-STRICT: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-STRICT-RND: call float @llvm.experimental.constrained.frem.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-DEFAULT: frem float
// LLVM-DEFAULT-RND: frem float
}

// LLVM-STRICT: attributes #[[$STRICT_ATTR]] = { {{.*}}strictfp{{.*}} }
// LLVM-STRICT-RND: attributes #[[$STRICT_ATTR]] = { {{.*}}strictfp{{.*}} }
// LLVM-DEFAULT-RND: attributes #[[$STRICT_ATTR]] = { {{.*}}strictfp{{.*}} }
