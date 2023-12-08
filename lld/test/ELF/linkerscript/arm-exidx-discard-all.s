// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple arm-gnu-linux-eabi -mcpu cortex-a7 -arm-add-build-attributes %s -o %t.o
// RUN: echo "ENTRY(__entrypoint) SECTIONS { /DISCARD/ : { *(.text.1) } }" > %t.script
// RUN: ld.lld -T %t.script %t.o -o %t.elf
// RUN: llvm-readobj --sections %t.elf | FileCheck %s

/// Test that when we /DISCARD/ all the input sections with associated
/// .ARM.exidx sections then we also discard all the .ARM.exidx sections.

 .section .text.1, "ax", %progbits
 .global foo
 .type foo, %function
 .fnstart
foo:
  bx lr
  .cantunwind
  .fnend

// CHECK-NOT: .ARM.exidx
