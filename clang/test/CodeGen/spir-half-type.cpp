// RUN: %clang_cc1 -O0 -triple spir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir64 -emit-llvm %s -o - | FileCheck %s

// This file tests that using the _Float16 type with the spir target will not
// use the llvm intrinsics but instead will use the half arithmetic
// instructions directly.

// Previously attempting to use a constant _Float16 with a comparison
// instruction when the target is spir or spir64 lead to an assert being hit.
bool fcmp_const() {
  _Float16 a = 0.0f16;
  const _Float16 b = 1.0f16;

  // CHECK-NOT: llvm.convert.to.fp16
  // CHECK-NOT: llvm.convert.from.fp16

  // CHECK: [[REG1:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp olt half [[REG1]], 0xH3C00

  // CHECK: [[REG2:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp olt half [[REG2]], 0xH4000

  // CHECK: [[REG3:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp ogt half [[REG3]], 0xH3C00

  // CHECK: [[REG4:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp ogt half [[REG4]], 0xH4200

  // CHECK: [[REG5:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp oeq half [[REG5]], 0xH3C00

  // CHECK: [[REG7:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp oeq half [[REG7]], 0xH4400

  // CHECK: [[REG8:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp une half [[REG8]], 0xH3C00

  // CHECK: [[REG9:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp une half [[REG9]], 0xH4500

  // CHECK: [[REG10:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp ole half [[REG10]], 0xH3C00

  // CHECK: [[REG11:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp ole half [[REG11]], 0xH4600

  // CHECK: [[REG12:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp oge half [[REG12]], 0xH3C00

  // CHECK: [[REG13:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: fcmp oge half [[REG13]], 0xH4700
  return a < b || a < 2.0f16 || a > b || a > 3.0f16 || a == b || a == 4.0f16 ||
         a != b || a != 5.0f16 || a <= b || a <= 6.0f16 || a >= b ||
         a >= 7.0f16;
}

bool fcmp() {
  _Float16 a = 0.0f16;
  _Float16 b = 1.0f16;

  // CHECK-NOT: llvm.convert.to.fp16
  // CHECK-NOT: llvm.convert.from.fp16
  // CHECK: [[REG1:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG2:%.*]] = load half, ptr %b, align 2
  // CHECK-NEXT: fcmp olt half [[REG1]], [[REG2]]

  // CHECK: [[REG3:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG4:%.*]] = load half, ptr %b, align 2
  // CHECK-NEXT: fcmp ogt half [[REG3]], [[REG4]]

  // CHECK: [[REG5:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG6:%.*]] = load half, ptr %b, align 2
  // CHECK-NEXT: fcmp oeq half [[REG5]], [[REG6]]

  // CHECK: [[REG7:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG8:%.*]] = load half, ptr %b, align 2
  // CHECK-NEXT: fcmp une half [[REG7]], [[REG8]]

  // CHECK: [[REG7:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG8:%.*]] = load half, ptr %b, align 2
  // CHECK-NEXT: fcmp ole half [[REG7]], [[REG8]]

  // CHECK: [[REG7:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG8:%.*]] = load half, ptr %b, align 2
  // CHECK-NEXT: fcmp oge half [[REG7]], [[REG8]]
  return a < b || a > b || a == b || a != b || a <= b || a >= b;
}

_Float16 fadd() {
  _Float16 a = 1.0f16;
  const _Float16 b = 2.0f16;

  // CHECK-NOT: llvm.convert.to.fp16
  // CHECK-NOT: llvm.convert.from.fp16

  // CHECK: [[REG1:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG2:%.*]] = fadd half [[REG1]], 0xH4000
  // CHECK-NEXT: [[REG3:%.*]] = fadd half [[REG2]], 0xH4200
  // CHECK-NEXT: ret half [[REG3]]
  return a + b + 3.0f16;
}

_Float16 fsub() {
  _Float16 a = 1.0f16;
  const _Float16 b = 2.0f16;

  // CHECK-NOT: llvm.convert.to.fp16
  // CHECK-NOT: llvm.convert.from.fp16

  // CHECK: [[REG1:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG2:%.*]] = fsub half [[REG1]], 0xH4000
  // CHECK-NEXT: [[REG3:%.*]] = fsub half [[REG2]], 0xH4200
  // CHECK-NEXT: ret half [[REG3]]
  return a - b - 3.0f16;
}

// CHECK: define{{.*}} spir_func noundef half @_Z4fmulDF16_(half noundef %arg)
_Float16 fmul(_Float16 arg) {
  _Float16 a = 1.0f16;
  const _Float16 b = 2.0f16;

  // CHECK-NOT: llvm.convert.to.fp16
  // CHECK-NOT: llvm.convert.from.fp16

  // CHECK: [[REG1:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG2:%.*]] = load half, ptr %arg.addr, align 2
  // CHECK-NEXT: [[REG3:%.*]] = fmul half [[REG1]], [[REG2]]
  // CHECK-NEXT: [[REG4:%.*]] = fmul half [[REG3]], 0xH4000
  // CHECK-NEXT: [[REG5:%.*]] = fmul half [[REG4]], 0xH4200
  // CHECK-NEXT: ret half [[REG5]]
  return a * arg * b * 3.0f16;
}

_Float16 fdiv() {
  _Float16 a = 1.0f16;
  const _Float16 b = 2.0f16;

  // CHECK-NOT: llvm.convert.to.fp16
  // CHECK-NOT: llvm.convert.from.fp16

  // CHECK: [[REG1:%.*]] = load half, ptr %a, align 2
  // CHECK-NEXT: [[REG2:%.*]] = fdiv half [[REG1]], 0xH4000
  // CHECK-NEXT: [[REG3:%.*]] = fdiv half [[REG2]], 0xH4200
  // CHECK-NEXT: ret half [[REG3]]
  return a / b / 3.0f16;
}
