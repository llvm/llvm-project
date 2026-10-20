# REQUIRES: loongarch
## Relax R_LARCH_CALL36. This test tests boundary cases and some special symbols.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax a.s -o a.o

# RUN: ld.lld -T lds a.o -o a
# RUN: llvm-objdump -d --no-show-raw-insn a | FileCheck %s --check-prefixes=RELAX,RELAX-MID

## Unsure whether this needs a diagnostic. GNU ld allows this.
# RUN: ld.lld -T lds -pie a.o -o a.pie
# RUN: llvm-objdump -d --no-show-raw-insn a.pie | FileCheck %s --check-prefixes=RELAX,RELAX-MID

# RUN: ld.lld -T lds -pie -z notext -z ifunc-noplt a.o -o a.ifunc-noplt
# RUN: llvm-objdump -d --no-show-raw-insn a.ifunc-noplt | FileCheck %s --check-prefixes=RELAX,NORELAX-MID

# RELAX-LABEL:  <_start>:
## offset = 0x10000000 - 0x8000000 = 0x8000000(134217728), hi=512, lo18=0
# RELAX-NEXT:    8000000:  pcaddu18i $ra, 512
# RELAX-NEXT:              jirl   $ra, $ra, 0
# RELAX-NEXT:              bl     134217720
# RELAX-NEXT:              bl     -134217728
## offset = 12 - 0x8000010 = -0x8000004(-134217732), hi=512, lo18=-4
# RELAX-NEXT:    8000010:  pcaddu18i $ra, -512
# RELAX-NEXT:              jirl   $ra, $ra, -4
# RELAX-EMPTY:

# RELAX-MID-LABEL:  <.mid>:
## offset = 0x8010000 - 0x8008000 = 32768
# RELAX-MID-NEXT:    8008000:  bl     32768
# RELAX-MID-NEXT:              b      32764
# RELAX-MID-EMPTY:

# NORELAX-MID-LABEL: <.mid>:
# NORELAX-MID-NEXT:  8008000:  pcaddu18i $ra, 0
# NORELAX-MID-NEXT:            jirl   $ra, $ra, 0
# NORELAX-MID-NEXT:            pcaddu18i $t0, 0
# NORELAX-MID-NEXT:            jr     $t0
# NORELAX-MID-EMPTY:

#--- a.s
.global _start, ifunc
_start:
  call36 pos       # exceed positive range (.text+0x7fffffc), not relaxed
  call36 pos       # relaxed
  call36 neg       # relaxed
  call36 neg       # exceed negative range (.text+16-0x8000000), not relaxed

.section .mid,"ax",@progbits
.balign 16
  call36 ifunc      # enable ifunc, not relaxed
  tail36 $t0, ifunc # enable ifunc, not relaxed

.type ifunc, @gnu_indirect_function
ifunc:
  ret

#--- lds
SECTIONS {
  .text 0x8000000 : { *(.text) }
  .mid  0x8008000 : { *(.mid) }
  .iplt 0x8010000 : { *(.iplt) }
}
neg = 12;
pos = 0x10000000;
