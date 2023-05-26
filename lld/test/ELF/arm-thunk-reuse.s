# REQUIRES: arm
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=armv7-a-none-eabi --arm-add-build-attributes %t/a.s -o %t/a.o
# RUN: ld.lld -pie -T %t/lds %t/a.o -o %t/a
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/a | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=armv7eb-a-none-eabi --arm-add-build-attributes -mcpu=cortex-a8 %t/a.s -o %t/a.o
# RUN: ld.lld -pie -T %t/lds %t/a.o -o %t/a
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/a | FileCheck %s

## We create a thunk for dest.
# CHECK-LABEL: <mid>:
# CHECK-NEXT:   2010004:     b       0x2010008 <__ARMV7PILongThunk_dest>
# CHECK-EMPTY:
# CHECK-NEXT:  <__ARMV7PILongThunk_dest>:
# CHECK-NEXT:   2010008:     movw    r12, #65516
# CHECK-NEXT:                movt    r12, #65023
# CHECK-NEXT:                add     r12, r12, pc
# CHECK-NEXT:                bx      r12

## The first instruction can reuse the thunk but the second can't.
## If we reuse the thunk for b, we will get an "out of range" error.
# CHECK-LABEL: <high>:
# CHECK-NEXT:   4010000:      bl      0x2010008 <__ARMV7PILongThunk_dest>
# CHECK-NEXT:                 b       0x4010008 <__ARMV7PILongThunk_dest>
# CHECK-EMPTY:
# CHECK-NEXT:  <__ARMV7PILongThunk_dest>:
# CHECK-NEXT:   4010008:      movw    r12, #65516
# CHECK-NEXT:                 movt    r12, #64511
# CHECK-NEXT:                 add     r12, r12, pc
# CHECK-NEXT:                 bx      r12

#--- a.s
.section .text_low, "ax", %progbits

.globl _start
_start:
  nop
dest:
  bx lr

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
  .text_mid 0x2010004 : { *(.text_mid) }
  .text_high 0x4010000 : { *(.text_high) }
}
