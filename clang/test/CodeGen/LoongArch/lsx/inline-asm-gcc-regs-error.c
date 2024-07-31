// RUN: not %clang_cc1 -triple loongarch64 -emit-llvm -O2 %s 2>&1 -o - | FileCheck %s

typedef signed char v16i8 __attribute__((vector_size(16), aligned(16)));

void test() {
// CHECK: :[[#@LINE+1]]:28: error: unknown register name 'vr0' in asm
    register v16i8 p0 asm ("vr0");
// CHECK: :[[#@LINE+1]]:29: error: unknown register name '$vr32' in asm
    register v16i8 p32 asm ("$vr32");
}
