// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
// RUN: ld.lld -Ttext=0x12000 -defsym long=0x10000000 -defsym short=0x8012004 -defsym short2=0x8012008 -defsym short3=0x801200c %t -o %t.exe
// RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck %s

/// The AArch64AbsLongThunk requires 8-byte alignment just in case unaligned
/// accesses are disabled. This increases the thunk section alignment to 8,
/// and the alignment of the AArch64AbsLongThunk to 8. The short thunk form
/// can still use 4-byte alignment.
.text
.type _start, %function
.globl _start
_start:
 b short
 b short2
 b short3
 b long
 nop

// CHECK-LABEL: <_start>:
// CHECK-NEXT: 12000: b       0x12018 <__AArch64AbsLongThunk_short>
// CHECK-NEXT:        b       0x1201c <__AArch64AbsLongThunk_short2>
// CHECK-NEXT:        b       0x12020 <__AArch64AbsLongThunk_short3>
// CHECK-NEXT:        b       0x12028 <__AArch64AbsLongThunk_long>
// CHECK-NEXT:        nop
// CHECK-NEXT:        udf     #0x0
// CHECK-EMPTY:
// CHECK-LABEL: <__AArch64AbsLongThunk_short>:
// CHECK-NEXT: 12018: b       0x8012004 <__AArch64AbsLongThunk_long+0x7ffffdc>
// CHECK-EMPY:
// CHECK-LABEL: <__AArch64AbsLongThunk_short2>:
// CHECK-NEXT: 1201c: b       0x8012008 <__AArch64AbsLongThunk_long+0x7ffffe0>
// CHECK-EMPTY:
// CHECK-LABEL: <__AArch64AbsLongThunk_short3>:
// CHECK-NEXT: 12020: b       0x801200c <__AArch64AbsLongThunk_long+0x7ffffe4>
// CHECK-NEXT:        udf     #0x0
// CHECK-EMPTY:
// CHECK-LABEL: <__AArch64AbsLongThunk_long>:
// CHECK-NEXT: 12028: ldr     x16, 0x12030 <__AArch64AbsLongThunk_long+0x8>
// CHECK-NEXT:        br      x16
// CHECK-NEXT: 00 00 00 10   .word   0x10000000
// CHECK-NEXT: 00 00 00 00   .word   0x00000000
