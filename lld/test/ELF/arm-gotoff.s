// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabi %s -o %t.o
// RUN: ld.lld -z separate-loadable-segments %t.o -o %t
// RUN: llvm-readelf -S -r --symbols %t | FileCheck %s
// RUN: llvm-objdump --triple=armv7a-linux-gnueabi -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s

// Test the R_ARM_GOTOFF32 relocation

// CHECK:      [Nr] Name              Type            Address  Off    Size   ES Flg Lk Inf Al
// CHECK-NEXT: [ 0]                   NULL            00000000 000000 000000 00      0   0  0
// CHECK-NEXT: [ 1] .text             PROGBITS        00020000 010000 000010 00  AX  0   0  4
// CHECK-NEXT: [ 2] .got              PROGBITS        00030000 020000 000000 00  WA  0   0  4
// CHECK-NEXT: [ 3] .relro_padding    NOBITS          00030000 020000 000000 00  WA  0   0  1
// CHECK-NEXT: [ 4] .bss              NOBITS          00030000 020000 000014 00  WA  0   0  1

// CHECK:      00030000    10 OBJECT  GLOBAL DEFAULT    4 bar
// CHECK-NEXT: 0003000a    10 OBJECT  GLOBAL DEFAULT    4 obj

// DISASM:      <_start>:
// DISASM-NEXT:   bx lr
// Offset 0 from .got = bar
// DISASM:        .word 0x00000000
// Offset 10 from .got = obj
// DISASM-NEXT:   .word 0x0000000a
// Offset 15 from .got = obj +5
// DISASM-NEXT:   .word 0x0000000f
 .syntax unified
 .globl _start
_start:
 bx lr
 .word bar(GOTOFF)
 .word obj(GOTOFF)
 .word obj(GOTOFF)+5
 .type bar, %object
 .comm bar, 10
 .type obj, %object
 .comm obj, 10
