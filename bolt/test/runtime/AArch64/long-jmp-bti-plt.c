/* This test checks that LongJmp can (not) add BTI instructions to PLT entries
   when targeting them using stubs.

// REQUIRES: system-linux

// RUN: split-file %s %t

// RUN: %clang --target=aarch64-unknown-linux-gnu -mbranch-protection=standard \
// RUN: -no-pie %t/bti-plt.c -o %t.exe -Wl,-q -fuse-ld=lld \
// RUN: -Wl,-T,%t/link.ld  -Wl,-z,force-bti
// RUN: not llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s
// CHECK: BOLT-INFO: binary is using BTI
// CHECK: BOLT-ERROR: Cannot add BTI landing pad to ignored function abort@PLT

#--- link.ld

SECTIONS {
  .plt : { *(.plt .plt.*) }
} INSERT BEFORE .text;

#--- bti-plt.c

//*/

#include <stdio.h>

int main(void) {
  printf("Hello, World!\n");
  return 0;
}

// .data big enough so the new code placed after it has to use stubs to reach
// PLTs:
__asm__(".section .data\n"
        ".globl space\n"
        "space:\n"
        ".fill 0x8000000,1,0x0\n"
        ".text\n");
