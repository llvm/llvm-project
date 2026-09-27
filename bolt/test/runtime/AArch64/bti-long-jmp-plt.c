// This test checks that LongJmp can add BTI instructions to PLT entries when
// targeting them using stubs.
//
//  The test uses a non-default layout, where the .plt section is placed before
//  the text section. This is needed so the new code placed under the .data
//  section is far enough from the PLt to trigger shortJmp stubs.
//
// In practice, the PLT section can also be placed after the original text
// section. In this scenario, the text section has to be large enough to trigger
// shortJmp stubs.

// REQUIRES: system-linux

// RUN: split-file %s %t

// RUN: %clang --target=aarch64-unknown-linux-gnu -mbranch-protection=standard \
// RUN: -no-pie %t/bti-plt.c -o %t.exe -Wl,-q -fuse-ld=lld \
// RUN: -Wl,-T,%t/link.ld  -Wl,-z,force-bti
// RUN: llvm-bolt %t.exe -o %t.bolt | FileCheck %s
// CHECK: BOLT-INFO: binary is using BTI

// Checking PLT entries before running BOLT
// RUN: llvm-objdump -d -j .plt %t.exe | FileCheck %s --check-prefix=CHECK-EXE
// CHECK-EXE: <abort@plt>
// CHECK-EXE-NEXT: adrp    x16, {{0x[0-9a-f]+}}
// CHECK-EXE-NEXT: ldr     x17, [x16, #{{0x[0-9a-f]+}}]
// CHECK-EXE-NEXT: add     x16, x16, #{{0x[0-9a-f]+}}
// CHECK-EXE-NEXT: br      x17
// CHECK-EXE-NEXT: nop
// CHECK-EXE-NEXT: nop

// Checking PLT entries after patching them in BOLT
// RUN: llvm-objdump -d -j .plt %t.bolt | FileCheck %s \
// RUN: --check-prefix=CHECK-BOLT
// CHECK-BOLT: <abort@plt>
// CHECK-BOLT-NEXT: bti     c
// CHECK-BOLT-NEXT: adrp    x16, {{0x[0-9a-f]+}}
// CHECK-BOLT-NEXT: ldr     x17, [x16, #{{0x[0-9a-f]+}}]
// CHECK-BOLT-NEXT: add     x16, x16, #{{0x[0-9a-f]+}}
// CHECK-BOLT-NEXT: br      x17
// CHECK-BOLT-NEXT: nop

/*
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
