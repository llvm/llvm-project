# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: ld.lld -pie -T %t/lds %t/a.o -o %t/a
# RUN: llvm-objdump -d --no-show-raw-insn %t/a | FileCheck %s

## We create a thunk for dest.
# CHECK-LABEL: <mid>:
# CHECK-NEXT:   8010008:       b       0x801000c <__AArch64ADRPThunk_>
# CHECK-EMPTY:
# CHECK-NEXT:  <__AArch64ADRPThunk_>:
# CHECK-NEXT:   801000c:       adrp    x16, 0x10000
# CHECK-NEXT:                  add     x16, x16, #4
# CHECK-NEXT:                  br      x16

## The first instruction can reuse the thunk but the second can't.
## If we reuse the thunk for b, we will get an "out of range" error.
# CHECK-LABEL: <high>:
# CHECK-NEXT:  1001000c:       bl      0x801000c <__AArch64ADRPThunk_>
# CHECK-NEXT:                  b       0x10010014 <__AArch64ADRPThunk_>
# CHECK-EMPTY:
# CHECK-NEXT:  <__AArch64ADRPThunk_>:
# CHECK-NEXT:  10010014:       adrp    x16, 0x10000
# CHECK-NEXT:                  add     x16, x16, #4
# CHECK-NEXT:                  br      x16

#--- a.s
.section .text_low, "ax", %progbits
.globl _start
_start:
  nop
dest:
  ret

.section .text_mid, "ax", %progbits
mid:
  b dest

.section .text_high, "ax", %progbits
high:
  bl dest
  b dest

#--- lds
SECTIONS {
  .text_low 0x10000: { *(.text_low) }
  .text_mid 0x8010008 : { *(.text_mid) }
  .text_high 0x1001000c : { *(.text_high) }
}
