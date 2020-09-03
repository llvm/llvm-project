// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:       .text_low 0x2000: { *(.text_low) } \
// RUN:       .text_high 0x8002000 : { *(.text_high) } \
// RUN:       } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s
// RUN: llvm-nm --no-sort --special-syms %t | FileCheck --check-prefix=NM %s

// Check that we have the out of branch range calculation right. The immediate
// field is signed so we have a slightly higher negative displacement.
 .section .text_low, "ax", %progbits
 .globl _start
 .type _start, %function
_start:
 // Need thunk to high_target@plt
 bl high_target
 // Need thunk to .text_high+4
 bl .text_high+4
 ret

 .section .text_high, "ax", %progbits
 .globl high_target
 .type high_target, %function
high_target:
 // No Thunk needed as we are within signed immediate range
 bl _start
 ret

// CHECK: Disassembly of section .text_low:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:     2000:       bl      0x200c <__AArch64AbsLongThunk_high_target>
// CHECK-NEXT:     2004:       bl      0x201c <__AArch64AbsLongThunk_>
// CHECK-NEXT:                 ret
// CHECK: <__AArch64AbsLongThunk_high_target>:
// CHECK-NEXT:     200c:       ldr     x16, 0x2014
// CHECK-NEXT:                 br      x16
// CHECK: <$d>:
// CHECK-NEXT:     2014:       00 20 00 08     .word   0x08002000
// CHECK-NEXT:     2018:       00 00 00 00     .word   0x00000000
// CHECK:      <__AArch64AbsLongThunk_>:
// CHECK-NEXT:     201c:       ldr x16, 0x2024
// CHECK-NEXT:     2020:       br x16
// CHECK:      <$d>:
// CHECK-NEXT:     2024:       04 20 00 08     .word   0x08002004
// CHECK-NEXT:     2028:       00 00 00 00     .word   0x00000000
// CHECK: Disassembly of section .text_high:
// CHECK-EMPTY:
// CHECK-NEXT: <high_target>:
// CHECK-NEXT:  8002000:       bl      0x2000 <_start>
// CHECK-NEXT:                 ret

/// Local symbols copied from %t.o
// NM:      t $x.0
// NM-NEXT: t $x.1
/// Local thunk symbols.
// NM-NEXT: t __AArch64AbsLongThunk_high_target
// NM-NEXT: t $x
// NM-NEXT: t $d
// NM-NEXT: t __AArch64AbsLongThunk_{{$}}
// NM-NEXT: t $x
// NM-NEXT: t $d
/// Global symbols.
// NM-NEXT: T _start
// NM-NEXT: T high_target
