# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc --filetype=obj -triple=loongarch64 %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/a.ld %t/a.o -o /dev/null

## This test ensures that the relaxation pass does not crash when it encounters
## a synthetic section (like .got) that has been placed inside an executable
## output section via a linker script. Synthetic sections do not have relaxAux
## data structures initialized.

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
