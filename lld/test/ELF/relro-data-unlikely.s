# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

# RUN: echo "SECTIONS { \
# RUN:  .data.rel.ro : { .data.rel.ro } \
# RUN:  .data.rel.ro.unlikely : { *(.data.rel.ro.unlikely) } \
# RUN: } INSERT AFTER .text " > %t.script

# RUN: ld.lld --script=%t.script %t.o -o %t
# RUN: llvm-readelf -l %t | FileCheck --check-prefix=SEG %s
# RUN: llvm-readelf -S %t | FileCheck %s

# There are 2 RW PT_LOAD segments. p_offset p_vaddr p_paddr p_filesz of the first
# should match PT_GNU_RELRO.
# The .data.rel.ro.unlikely section is in PT_GNU_RELRO segment.

#           Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# SEG:      LOAD           0x0001c8 0x00000000002011c8 0x00000000002011c8 0x000001 0x000001 R E 0x1000
# SEG-NEXT: LOAD           0x0001c9 0x00000000002021c9 0x00000000002021c9 0x001001 0x001e37 RW  0x1000
# SEG-NEXT: LOAD           0x0011ca 0x00000000002041ca 0x00000000002041ca 0x000001 0x000002 RW  0x1000
# SEG-NEXT: GNU_RELRO      0x0001c9 0x00000000002021c9 0x00000000002021c9 0x001001 0x001e37 R   0x1
# SEG-NEXT: GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x0

# SEG:      .text
# SEG-NEXT: .data.rel.ro .data.rel.ro.unlikely .relro_padding
# SEG-NEXT: .data .bss

#        [Nr] Name                    Type            Address          Off    Size
# CHECK:      .data.rel.ro            PROGBITS        00000000002021c9 0001c9 000001
# CHECK-NEXT: .data.rel.ro.unlikely   PROGBITS        00000000002021ca 0001ca 001000
# CHECK-NEXT: .relro_padding          NOBITS          00000000002031ca 0011ca 000e36
# CHECK-NEXT: .data                   PROGBITS        00000000002041ca 0011ca 000001
# CHECK-NEXT: .bss                    NOBITS          00000000002041cb 0011cb 000001

.globl _start
_start:
  ret


.section .data.rel.ro, "aw"
.space 1

.section .data.rel.ro.unlikely, "aw"
.space 4096

.section .data, "aw"
.space 1

.section .bss, "aw"
.space 1
