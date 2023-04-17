# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld -pie -Ttext=0x10300 %t.o -o %t
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --disassemble-symbols=dest,__AArch64ADRPThunk_,high %t | FileCheck %s

## We create initial ThunkSection before the gap. Because the ThunkSection
## selection code isn't so precise, we may create an unused thunk there (0x10704).
## In the next pass we will create a ThunkSection after the gap. There used to be
## a bug reusing the first ThunkSection (unreachable) due to the large r_addend.
# CHECK:       <dest>:
# CHECK-NEXT:     10700:       ret
# CHECK:       <__AArch64ADRPThunk_>:
# CHECK-NEXT:     10704:       b       0x10700 <dest>
# CHECK-EMPTY:
# CHECK:       <__AArch64ADRPThunk_>:
# CHECK-NEXT:   8010708:       adrp    x16, 0x10000
# CHECK-NEXT:                  add     x16, x16, #1792
# CHECk-NEXT:                  br      x16
# CHECK-LABEL: <high>:
# CHECK-NEXT:   8010714:       bl      0x8010708 <__AArch64ADRPThunk_>
# CHECK-NEXT:                  b       0x8010708 <__AArch64ADRPThunk_>

.section .text._start, "ax", %progbits
.globl _start
_start:
.space 0x400
dest:
  ret

.section .text.gap, "ax", %progbits
.space 0x8000000

.section .text.high, "ax", %progbits
high:
  bl dest
  b dest
