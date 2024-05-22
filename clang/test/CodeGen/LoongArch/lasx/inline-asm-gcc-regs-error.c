// RUN: not %clang_cc1 -triple loongarch64 -emit-llvm -O2 %s 2>&1 -o - | FileCheck %s

typedef signed char v32i8 __attribute__((vector_size(32), aligned(32)));

void test() {
// CHECK: :[[#@LINE+1]]:28: error: unknown register name 'xr0' in asm
    register v32i8 p0 asm ("xr0");
// CHECK: :[[#@LINE+1]]:29: error: unknown register name '$xr32' in asm
    register v32i8 p32 asm ("$xr32");
}
