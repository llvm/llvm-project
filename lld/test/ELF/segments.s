# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld a.o -o a
# RUN: llvm-readelf -l a | FileCheck --check-prefix=ROSEGMENT %s
# RUN: ld.lld --no-rosegment --rosegment a.o -o - | cmp - a
# RUN: ld.lld --omagic --no-omagic a.o -o - | cmp - a

# ROSEGMENT:       Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# ROSEGMENT-NEXT:  PHDR           0x000040 0x0000000000200040 0x0000000000200040 0x000118 0x000118 R   0x8
# ROSEGMENT-NEXT:  LOAD           0x000000 0x0000000000200000 0x0000000000200000 0x00015b 0x00015b R   0x1000
# ROSEGMENT-NEXT:  LOAD           0x00015c 0x000000000020115c 0x000000000020115c 0x000003 0x000003 R E 0x1000
# ROSEGMENT-NEXT:  LOAD           0x00015f 0x000000000020215f 0x000000000020215f 0x000002 0x000002 RW  0x1000
# ROSEGMENT-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# RUN: ld.lld --rosegment a.o -T a.lds -o ro1
# RUN: llvm-readelf -l ro1 | FileCheck --check-prefix=ROSEGMENT1 %s

# ROSEGMENT1:       Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# ROSEGMENT1-NEXT:  LOAD           0x001000 0x0000000000000000 0x0000000000000000 0x000001 0x000001 R   0x1000
# ROSEGMENT1-NEXT:  LOAD           0x001004 0x0000000000000004 0x0000000000000004 0x000002 0x000002 R E 0x1000
# ROSEGMENT1-NEXT:  LOAD           0x001006 0x0000000000000006 0x0000000000000006 0x000001 0x000001 RW  0x1000
# ROSEGMENT1-NEXT:  LOAD           0x001007 0x0000000000000007 0x0000000000000007 0x000001 0x000001 R E 0x1000
# ROSEGMENT1-NEXT:  LOAD           0x001008 0x0000000000000008 0x0000000000000008 0x000001 0x000001 R   0x1000
# ROSEGMENT1-NEXT:  LOAD           0x001009 0x0000000000000009 0x0000000000000009 0x000001 0x000001 RW  0x1000
# ROSEGMENT1-NEXT:  LOAD           0x00100a 0x000000000000000a 0x000000000000000a 0x000001 0x000001 R   0x1000
# ROSEGMENT1-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# RUN: ld.lld --no-rosegment a.o -o noro
# RUN: llvm-readelf -l noro | FileCheck --check-prefix=NOROSEGMENT %s

# NOROSEGMENT:       Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# NOROSEGMENT-NEXT:  PHDR           0x000040 0x0000000000200040 0x0000000000200040 0x0000e0 0x0000e0 R   0x8
# NOROSEGMENT-NEXT:  LOAD           0x000000 0x0000000000200000 0x0000000000200000 0x000127 0x000127 R E 0x1000
# NOROSEGMENT-NEXT:  LOAD           0x000127 0x0000000000201127 0x0000000000201127 0x000002 0x000002 RW  0x1000
# NOROSEGMENT-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# RUN: ld.lld --no-rosegment a.o -T a.lds -o noro1
# RUN: llvm-readelf -l noro1 | FileCheck --check-prefix=NOROSEGMENT1 %s

# NOROSEGMENT1:       Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# NOROSEGMENT1-NEXT:  LOAD           0x001000 0x0000000000000000 0x0000000000000000 0x000006 0x000006 R E 0x1000
# NOROSEGMENT1-NEXT:  LOAD           0x001006 0x0000000000000006 0x0000000000000006 0x000001 0x000001 RW  0x1000
# NOROSEGMENT1-NEXT:  LOAD           0x001007 0x0000000000000007 0x0000000000000007 0x000002 0x000002 R E 0x1000
# NOROSEGMENT1-NEXT:  LOAD           0x001009 0x0000000000000009 0x0000000000000009 0x000001 0x000001 RW  0x1000
# NOROSEGMENT1-NEXT:  LOAD           0x00100a 0x000000000000000a 0x000000000000000a 0x000001 0x000001 R   0x1000
# NOROSEGMENT1-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# RUN: ld.lld -N a.o -o omagic
# RUN: llvm-readelf -l omagic | FileCheck --check-prefix=OMAGIC %s
# RUN: ld.lld --omagic a.o -o - | cmp - omagic

# OMAGIC:       Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# OMAGIC-NEXT:  LOAD           0x0000b0 0x00000000002000b0 0x00000000002000b0 0x000009 0x000009 RWE 0x4
# OMAGIC-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# RUN: ld.lld -n a.o -o nmagic
# RUN: llvm-readelf -l nmagic | FileCheck --check-prefix=NMAGIC %s
# RUN: ld.lld --nmagic a.o -o - | cmp nmagic -

# NMAGIC:       Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# NMAGIC-NEXT:  LOAD           0x000120 0x0000000000200120 0x0000000000200120 0x000003 0x000003 R   0x1
# NMAGIC-NEXT:  LOAD           0x000124 0x0000000000200124 0x0000000000200124 0x000003 0x000003 R E 0x4
# NMAGIC-NEXT:  LOAD           0x000127 0x0000000000200127 0x0000000000200127 0x000002 0x000002 RW  0x1
# NMAGIC-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

#--- a.s
.global _start
_start:
 nop

.section .ro1,"a"; .byte 1
.section .rw1,"aw"; .byte 3
.section .rx1,"ax"; .byte 2

.section .ro2,"a"; .byte 1
.section .rw2,"aw"; .byte 3
.section .rx2,"ax"; .byte 2

.section .ro3,"a"; .byte 1

#--- a.lds
SECTIONS {
  .ro1 : {}
  .text : {}
  .rx1 : {}
  .rw1 : {}
  .rx2 : {}
  .ro2 : {}
  .rw2 : {}
  .ro3 : {}
}
