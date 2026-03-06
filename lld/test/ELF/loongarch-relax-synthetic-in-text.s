# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc --filetype=obj -triple=loongarch64 -mattr=+relax %t/a.s -o %t/a.o

## Do not crash when we encounter a synthetic section (like .got) that has
## been placed inside an executable output section via a linker script.
## Synthetic sections do not have relaxAux data structures initialized.

# RUN: ld.lld -T %t/a.ld %t/a.o -o %t/nopie
# RUN: llvm-objdump -s -j.text %t/nopie | FileCheck %s --check-prefixes=CHECK-NOPIE

# RUN: ld.lld -pie -T %t/a.ld %t/a.o -o %t/pie
# RUN: llvm-objdump -s -j.text %t/pie | FileCheck %s --check-prefix=CHECK-PIE

# CHECK-NOPIE:      Contents of section .text:
# CHECK-NOPIE-NEXT: 0000 0400001a 8440c002 10000000 00000000

# CHECK-PIE:        Contents of section .text:
# CHECK-PIE-NEXT:   0060 0400001a 8400c502 00000000 00000000

#--- a.s
.global _start
_start:
  pcalau12i $a0, %got_pc_hi20(sym)
  ld.d $a0, $a0, %got_pc_lo12(sym)

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
