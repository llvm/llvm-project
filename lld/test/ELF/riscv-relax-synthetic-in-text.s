# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc --filetype=obj -triple=riscv64 -mattr=+relax %t/a.s -o %t/a.o

## Do not crash when we encounter a synthetic section (like .got) that has
## been placed inside an executable output section via a linker script.
## Synthetic sections do not have relaxAux data structures initialized.

# RUN: ld.lld -T %t/a.ld %t/a.o -o %t/a.out
# RUN: llvm-objdump -s %t/a.out | FileCheck %s

# CHECK:      Contents of section .text:
# CHECK-NEXT: 17050000 03350501 00000000 00000000
# CHECK-NEXT: 18000000 00000000

#--- a.s
.global _start
_start:
1:
  auipc a0, %got_pcrel_hi(sym)
  ld a0, %pcrel_lo(1b)(a0)

.data
sym:
  .word 0

#--- a.ld
SECTIONS {
  .text : {
    *(.text)
    *(.got)
  }
}
