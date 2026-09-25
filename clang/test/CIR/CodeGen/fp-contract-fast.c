// -ffp-contract=fast does not form cir.fmuladd. It stamps `contract` on the
// individual floating-point ops so a backend in Standard fusion mode can
// still contract them, including across statements.
// -ffp-contract=on still forms cir.fmuladd and does not set `contract`.
// -ffp-contract=off does neither.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast -emit-cir %s -o %t-fast.cir
// RUN: FileCheck --input-file=%t-fast.cir %s -check-prefix=CIR-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast -emit-llvm %s -o %t-fast.ll
// RUN: FileCheck --input-file=%t-fast.ll %s -check-prefix=LLVM-FAST

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -emit-cir %s -o %t-on.cir
// RUN: FileCheck --input-file=%t-on.cir %s -check-prefix=CIR-ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=off -emit-cir %s -o %t-off.cir
// RUN: FileCheck --input-file=%t-off.cir %s -check-prefix=CIR-OFF

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=fast -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --input-file=%t-og.ll %s -check-prefix=LLVM-FAST

float same_stmt(float a, float b, float c) { return a * b + c; }
// CIR-FAST-LABEL: cir.func {{.*}}@same_stmt
// CIR-FAST: cir.fmul {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST-NOT: cir.fmuladd

// CIR-ON-LABEL: cir.func {{.*}}@same_stmt
// CIR-ON: cir.fmuladd
// CIR-ON-NOT: #cir.fastmath

// CIR-OFF-LABEL: cir.func {{.*}}@same_stmt
// CIR-OFF: cir.fmul {{.*}} : !cir.float
// CIR-OFF: cir.fadd {{.*}} : !cir.float
// CIR-OFF-NOT: #cir.fastmath
// CIR-OFF-NOT: cir.fmuladd

// LLVM-FAST-LABEL: @same_stmt
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float
// LLVM-FAST-NOT: @llvm.fmuladd

float across_stmt(float a, float b, float c) {
  float t = a * b;
  return t + c;
}
// CIR-FAST-LABEL: cir.func {{.*}}@across_stmt
// CIR-FAST: cir.fmul {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST-NOT: cir.fmuladd

// CIR-ON-LABEL: cir.func {{.*}}@across_stmt
// CIR-ON: cir.fmul {{.*}} : !cir.float
// CIR-ON: cir.fadd {{.*}} : !cir.float
// CIR-ON-NOT: cir.fmuladd
// CIR-ON-NOT: #cir.fastmath

// LLVM-FAST-LABEL: @across_stmt
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float

float sub_stmt(float a, float b, float c) { return a * b - c; }
// CIR-FAST-LABEL: cir.func {{.*}}@sub_stmt
// CIR-FAST: cir.fmul {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST: cir.fsub {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST-NOT: cir.fmuladd

// CIR-ON-LABEL: cir.func {{.*}}@sub_stmt
// CIR-ON: cir.fmuladd
// CIR-ON-NOT: #cir.fastmath

// CIR-OFF-LABEL: cir.func {{.*}}@sub_stmt
// CIR-OFF: cir.fmul {{.*}} : !cir.float
// CIR-OFF: cir.fsub {{.*}} : !cir.float
// CIR-OFF-NOT: #cir.fastmath
// CIR-OFF-NOT: cir.fmuladd

// LLVM-FAST-LABEL: @sub_stmt
// LLVM-FAST: fmul contract float
// LLVM-FAST: fsub contract float

float neg(float a) { return -a; }
// CIR-FAST-LABEL: cir.func {{.*}}@neg
// CIR-FAST: cir.fneg {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-ON-LABEL: cir.func {{.*}}@neg
// CIR-ON: cir.fneg {{.*}} : !cir.float
// CIR-ON-NOT: #cir.fastmath
// CIR-OFF-LABEL: cir.func {{.*}}@neg
// CIR-OFF: cir.fneg {{.*}} : !cir.float
// CIR-OFF-NOT: #cir.fastmath
// LLVM-FAST-LABEL: @neg
// LLVM-FAST: fneg contract float

int cmp(float a, float b) { return a < b; }
// CIR-FAST-LABEL: cir.func {{.*}}@cmp
// CIR-FAST: cir.cmp lt {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-ON-LABEL: cir.func {{.*}}@cmp
// CIR-ON: cir.cmp lt {{.*}} : !cir.float
// CIR-ON-NOT: #cir.fastmath
// CIR-OFF-LABEL: cir.func {{.*}}@cmp
// CIR-OFF: cir.cmp lt {{.*}} : !cir.float
// CIR-OFF-NOT: #cir.fastmath
// LLVM-FAST-LABEL: @cmp
// LLVM-FAST: fcmp contract olt float

float rem(float a, float b) { return __builtin_fmodf(a, b); }
// CIR-FAST-LABEL: cir.func {{.*}}@rem
// CIR-FAST: cir.fmod {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-ON-LABEL: cir.func {{.*}}@rem
// CIR-ON: cir.fmod {{.*}} : !cir.float
// CIR-ON-NOT: #cir.fastmath
// CIR-OFF-LABEL: cir.func {{.*}}@rem
// CIR-OFF: cir.fmod {{.*}} : !cir.float
// CIR-OFF-NOT: #cir.fastmath
// LLVM-FAST-LABEL: @rem
// LLVM-FAST: frem contract float

float sq(float a) { return __builtin_sqrtf(a); }
// CIR-FAST-LABEL: cir.func {{.*}}@sq
// CIR-FAST: cir.sqrt {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-ON-LABEL: cir.func {{.*}}@sq
// CIR-ON: cir.sqrt {{.*}} : !cir.float
// CIR-ON-NOT: #cir.fastmath
// CIR-OFF-LABEL: cir.func {{.*}}@sq
// CIR-OFF: cir.sqrt {{.*}} : !cir.float
// CIR-OFF-NOT: #cir.fastmath
// LLVM-FAST-LABEL: @sq
// LLVM-FAST: call contract float @llvm.sqrt.f32

float absf(float a) { return __builtin_fabsf(a); }
// CIR-FAST-LABEL: cir.func {{.*}}@absf
// CIR-FAST: cir.fabs {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-ON-LABEL: cir.func {{.*}}@absf
// CIR-ON: cir.fabs {{.*}} : !cir.float
// CIR-ON-NOT: #cir.fastmath
// CIR-OFF-LABEL: cir.func {{.*}}@absf
// CIR-OFF: cir.fabs {{.*}} : !cir.float
// CIR-OFF-NOT: #cir.fastmath
// LLVM-FAST-LABEL: @absf
// LLVM-FAST: call contract float @llvm.fabs.f32
