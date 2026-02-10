// Make sure that during the Hugify pass BOLT adds a BTI instruction to _start.

#include <stdio.h>

int main(int argc, char **argv) {
  printf("Hello world\n");
  return 0;
}

/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags -mbranch-protection=standard -no-pie %s -o %t.nopie.exe \
RUN: -Wl,-q,-z,force-bti
RUN: %clang %cflags -mbranch-protection=standard -fpic %s -o %t.pie.exe \
RUN: -Wl,-q,-z,force-bti

RUN: llvm-bolt %t.nopie.exe --lite=0 -o %t.nopie.bolt --hugify | FileCheck %s \
RUN: -check-prefix=CHECK-BOLT
RUN: llvm-bolt %t.pie.exe --lite=0 -o %t.pie.bolt --hugify | FileCheck %s \
RUN: -check-prefix=CHECK-BOLT

CHECK-BOLT: binary is using BTI

RUN: llvm-objdump -D %t.nopie.bolt | FileCheck %s -check-prefix=CHECK-OBJDUMP
RUN: llvm-objdump -D %t.pie.bolt | FileCheck %s -check-prefix=CHECK-OBJDUMP

CHECK-OBJDUMP: <_start>:
CHECK-OBJDUMP-NEXT: bti

CHECK-OBJDUMP: <__bolt_hugify_start_program>:
CHECK-OBJDUMP-NEXT: adrp x16, 0x[[#%x,ADRP:]]
CHECK-OBJDUMP-NEXT: add x16, x16, #0x[[#%x,ADD:]]
CHECK-OBJDUMP-NEXT: br x16

*/
