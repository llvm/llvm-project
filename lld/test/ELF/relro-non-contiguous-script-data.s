# REQUIRES: x86
## Script data commands (`. += 1` or BYTE) make an output section need a
## PT_LOAD, splitting the relro region. Emit one PT_GNU_RELRO per run.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o b.o
# RUN: ld.lld -shared -soname=b b.o -o b.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld a.o b.so -T 1.lds -o out1
# RUN: llvm-readelf -l out1 | FileCheck %s

# RUN: ld.lld a.o b.so -T 2.lds -o out2
# RUN: llvm-readelf -l out2 | FileCheck %s

# CHECK:      Type      Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK:      GNU_RELRO 0x001048 0x0000000000000048 0x0000000000000048 0x000090 0x000090 R   0x1
# CHECK-NEXT: GNU_RELRO 0x0010dc 0x00000000000000dc 0x00000000000000dc 0x000008 0x000008 R   0x1

#--- 1.lds
SECTIONS {
  .dynamic : { *(.dynamic) }
  .non_ro : { . += 1; }
  .jcr : { *(.jcr) }
}

#--- 2.lds
SECTIONS {
  .dynamic : { *(.dynamic) }
  .non_ro : { BYTE(1); }
  .jcr : { *(.jcr) }
}

#--- a.s
.global _start
_start:

## non-empty relro section
.section .jcr, "aw", @progbits
.p2align 2
.quad 0
