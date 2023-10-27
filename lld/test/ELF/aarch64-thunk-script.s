// REQUIRES: aarch64
// RUN: split-file %s %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 %t/asm -o %t.o
// RUN: ld.lld --script %t/lds %t.o -o %t/out
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t/out | FileCheck %s
// RUN: llvm-nm --no-sort --special-syms %t/out | FileCheck --check-prefix=NM %s

//--- asm
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
// CHECK-NEXT:     2004:       bl      0x2010 <__AArch64AbsLongThunk_>
// CHECK-NEXT:                 ret
// CHECK: <__AArch64AbsLongThunk_high_target>:
// CHECK-NEXT:     200c:       b       0x8002000 <high_target>
// CHECK: <__AArch64AbsLongThunk_>:
// CHECK-NEXT:     2010:       b       0x8002004 <high_target+0x4>
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
// NM-NEXT: t __AArch64AbsLongThunk_{{$}}
// NM-NEXT: t $x
/// Global symbols.
// NM-NEXT: T _start
// NM-NEXT: T high_target

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .text_low 0x2000: { *(.text_low) } :low
  .text_high 0x8002000 : { *(.text_high) } :high
}
