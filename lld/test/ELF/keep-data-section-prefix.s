# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld -z keep-data-section-prefix -T x.lds a.o -o out1
# RUN: llvm-readelf -l out1 | FileCheck --check-prefixes=SEG,LS %s
# RUN: llvm-readelf -S out1 | FileCheck %s --check-prefix=CHECK-LS

# RUN: ld.lld -z keep-data-section-prefix a.o -o out2
# RUN: llvm-readelf -l out2 | FileCheck --check-prefixes=SEG,PRE %s
# RUN: llvm-readelf -S out2 | FileCheck %s --check-prefix=CHECK-PRE

# RUN: ld.lld a.o -o out3
# RUN: llvm-readelf -l out3 | FileCheck --check-prefixes=SEG,PRE %s
# RUN: llvm-readelf -S out3 | FileCheck %s --check-prefix=CHECK-PRE

# RUN: not ld.lld -T x.lds a.o 2>&1 | FileCheck %s
# CHECK: error: section: .relro_padding is not contiguous with other relro sections

## The first RW PT_LOAD segment has FileSiz 0x126f (0x1000 + 0x200 + 0x60 + 0xf),
## and its p_offset p_vaddr p_paddr p_filesz should match PT_GNU_RELRO.
#           Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# SEG:      LOAD           0x0001c8 0x00000000002011c8 0x00000000002011c8 0x000001 0x000001 R E 0x1000
# SEG-NEXT: LOAD           0x0001c9 0x00000000002021c9 0x00000000002021c9 0x00126f 0x001e37 RW  0x1000
# SEG-NEXT: LOAD           0x001438 0x0000000000204438 0x0000000000204438 0x000001 0x000002 RW  0x1000
# SEG-NEXT: GNU_RELRO      0x0001c9 0x00000000002021c9 0x00000000002021c9 0x00126f 0x001e37 R   0x1
# SEG-NEXT: GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x0

## Input to output mapping per linker script
##   .data.rel.ro.split -> .data.rel.ro
##   .data.rel.ro -> .data.rel.ro
##   .data.rel.ro.hot -> .data.rel.ro.hot
##   .data.rel.ro.unlikely -> .data.rel.ro.unlikely
# LS:      .text
# LS-NEXT: .data.rel.ro.hot .data.rel.ro .data.rel.ro.unlikely .relro_padding
# LS-NEXT: .data .bss

#        [Nr] Name                    Type            Address          Off    Size
# CHECK-LS:      .data.rel.ro.hot        PROGBITS        00000000002021c9 0001c9 00000f
# CHECK-LS-NEXT: .data.rel.ro            PROGBITS        00000000002021d8 0001d8 000260
# CHECK-LS-NEXT: .data.rel.ro.unlikely   PROGBITS        0000000000202438 000438 001000
# CHECK-LS-NEXT: .relro_padding          NOBITS          0000000000203438 001438 000bc8
# CHECK-LS-NEXT: .data                   PROGBITS        0000000000204438 001438 000001
# CHECK-LS-NEXT: .bss                    NOBITS          0000000000204439 001439 000001

## Linker script is not provided to map data sections.
## So all input sections with prefix .data.rel.ro will map to .data.rel.ro in the output.
# PRE:      .text
# PRE-NEXT: .data.rel.ro .relro_padding
# PRE-NEXT: .data .bss

#        [Nr] Name                    Type            Address          Off    Size
# CHECK-PRE:      .data.rel.ro            PROGBITS        00000000002021c9 0001c9 00126f
# CHECK-PRE-NEXT: .relro_padding          NOBITS          0000000000203438 001438 000bc8
# CHECK-PRE-NEXT: .data                   PROGBITS        0000000000204438 001438 000001
# CHECK-PRE-NEXT: .bss                    NOBITS          0000000000204439 001439 000001

#--- x.lds
SECTIONS {
  .data.rel.ro.hot : { *(.data.rel.ro.hot) }
  .data.rel.ro : { .data.rel.ro }
  .data.rel.ro.unlikely : { *(.data.rel.ro.unlikely) }
} INSERT AFTER .text


#--- a.s
.globl _start
_start:
  ret

.section .data.rel.ro.hot, "aw"
.space 15

.section .data.rel.ro, "aw"
.space 96

.section .data.rel.ro.split,"aw"
.space 512

.section .data.rel.ro.unlikely, "aw"
.space 4096

.section .data, "aw"
.space 1

.section .bss, "aw"
.space 1
