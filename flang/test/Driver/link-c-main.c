/*
Test that an object file with a C main function can be linked to an executable
by Flang.

For now, this test only covers the Gnu toolchain on Linux.

REQUIRES: x86-registered-target || aarch64-registered-target || riscv64-registered-target
REQUIRES: system-linux, c-compiler

RUN: %cc -c %s -o %t.o
RUN: %flang -target x86_64-unknown-linux-gnu %t.o -o %t.out -flang-experimental-exec
RUN: llvm-objdump --syms %t.out | FileCheck %s --implicit-check-not Fortran

Test that it also works if the c-main is bundled in an archive.

RUN: llvm-ar -r %t.a %t.o
RUN: %flang -target x86_64-unknown-linux-gnu %t.a -o %ta.out -flang-experimental-exec
RUN: llvm-objdump --syms %ta.out | FileCheck %s --implicit-check-not Fortran
*/

int main(void) {
    return 0;
}

/*
CHECK-DAG: F .text {{[a-f0-9]+}} main
CHECK-DAG: *UND* {{[a-f0-9]+}} _QQmain
*/
