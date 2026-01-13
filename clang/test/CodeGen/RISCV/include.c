// RUN: %clang --target=riscv32-unknown-elf -I %S/../Inputs/ -flto %s -c -o %t.full.bc
// RUN: llvm-dis %t.full.bc -o - | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -I %S/../Inputs/ -flto=thin %s -c -o %t.thin.bc
// RUN: llvm-dis %t.thin.bc -o - | FileCheck %s
__asm__(".include \"macros.s\"");

void test() {
}

// CHECK: module asm ".include \22macros.s\22"

