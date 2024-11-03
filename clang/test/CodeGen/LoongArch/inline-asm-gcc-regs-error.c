// RUN: not %clang_cc1 -triple loongarch32 -emit-llvm %s 2>&1 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple loongarch64 -emit-llvm %s 2>&1 -o - | FileCheck %s

void test(void) {
// CHECK: :[[#@LINE+1]]:24: error: unknown register name '$r32' in asm
  register int a0 asm ("$r32");
// CHECK: :[[#@LINE+1]]:26: error: unknown register name '$f32' in asm
  register float a1 asm ("$f32");
// CHECK: :[[#@LINE+1]]:24: error: unknown register name '$foo' in asm
  register int a2 asm ("$foo");

/// Names not prefixed with '$' are invalid.

// CHECK: :[[#@LINE+1]]:26: error: unknown register name 'f0' in asm
  register float a5 asm ("f0");
// CHECK: :[[#@LINE+1]]:26: error: unknown register name 'fa0' in asm
  register float a6 asm ("fa0");
// CHECK: :[[#@LINE+1]]:15: error: unknown register name 'fcc0' in asm
  asm ("" ::: "fcc0");
}
