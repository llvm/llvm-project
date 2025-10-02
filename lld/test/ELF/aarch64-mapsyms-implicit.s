# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 -implicit-mapsyms %s -o %t.o
# RUN: ld.lld %t.o -z keep-text-section-prefix -o %t
# RUN: llvm-objdump -d --no-print-imm-hex --show-all-symbols %t | FileCheck %s

# CHECK:      <_start>:
# CHECK-NEXT:   nop
# CHECK-EMPTY:
# CHECK-NEXT: <$d>:
# CHECK-NEXT:   .word   0x0000002a
# CHECK-EMPTY:
# CHECK-NEXT: <$x>:
# CHECK-NEXT:   nop
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section .text.hot:
# CHECK-EMPTY:
# CHECK-NEXT: <.text.hot>:
# CHECK-NEXT:   nop
# CHECK-EMPTY:
# CHECK-NEXT: <$d>:
# CHECK-NEXT:   .word   0x0000002a
# CHECK-EMPTY:
# CHECK-NEXT: <$d>:
# CHECK-NEXT: <$x>:
# CHECK-NEXT:   udf     #42
# CHECK-EMPTY:
# CHECK-NEXT: <$x>:
# CHECK-NEXT:   nop

## Trailing data followed by a section starting with an instruction.
.section .text.1,"ax"
.globl _start
_start:
  nop
  .long 42
.section .text.2,"ax"
  nop

## Trailing data followed by a section starting with a data directive.
.section .text.hot.1,"ax"
  nop
  .long 42
.section .text.hot.2,"ax"
  .long 42
  nop
