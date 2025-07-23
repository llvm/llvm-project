# REQUIRES: x86
## -z keep-data-section-prefix separates static data sections with prefix
## .<section>.hot, .<section>.unlikely in the absence of a SECTIONS command.

# RUN: rm -rf %t && split-file %s %t && cd %t

## Test that lld knows .data.rel.ro.unlikely and .data.rel.ro.hot are relocatable
## read-only data sections.
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld -z keep-data-section-prefix -T x1.lds a.o -o out1
# RUN: llvm-readelf -l out1 | FileCheck --check-prefixes=SEG,LS %s
# RUN: llvm-readelf -S out1 | FileCheck %s --check-prefix=CHECK-LS

# RUN: ld.lld a.o -o out3
# RUN: llvm-readelf -l out3 | FileCheck --check-prefixes=SEG,PRE %s
# RUN: llvm-readelf -S out3 | FileCheck %s --check-prefix=CHECK-PRE

# RUN: not ld.lld -T x1.lds a.o 2>&1 | FileCheck %s
# CHECK: error: section: .relro_padding is not contiguous with other relro sections

## Test that lld can group data sections based on its hotness prefix.

# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o

# RUN: ld.lld b.o -o out1
# RUN: llvm-readelf -S out1 | FileCheck --check-prefix=BASIC %s
# RUN: ld.lld -z nokeep-text-section-prefix b.o -o out2
# RUN: cmp out1 out2

# RUN: ld.lld -z keep-data-section-prefix b.o -o out3
# RUN: llvm-readelf -S out3 | FileCheck --check-prefix=KEEP %s

## With a SECTIONS command, orphan sections are created verbatim.
## No grouping is performed for them.
# RUN: ld.lld -T x2.lds -z keep-data-section-prefix b.o -o out4
# RUN: llvm-readelf -S out4 | FileCheck --check-prefix=SCRIPT %s

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

# BASIC:         [Nr] Name              Type            {{.*}}   Size              
# BASIC:         [ 1] .text       
# BASIC-NEXT:    [ 2] .data.rel.ro      PROGBITS        {{.*}}   00000f
# BASIC-NEXT:    [ 3] .bss.rel.ro       NOBITS          {{.*}}   000008
# BASIC-NEXT:    [ 4] .relro_padding    NOBITS          {{.*}}   000e20
# BASIC-NEXT:    [ 5] .rodata           PROGBITS        {{.*}}   000006
# BASIC-NEXT:    [ 6] .data             PROGBITS        {{.*}}   000004
# BASIC-NEXT:    [ 7] .bss              NOBITS          {{.*}}  000004

# KEEP:       [Nr]  Name                    Type            {{.*}}    Size
# KEEP:       [ 1] .text
# KEEP-NEXT:  [ 2] .data.rel.ro             PROGBITS        {{.*}} 000009
# KEEP-NEXT:  [ 3] .data.rel.ro.hot         PROGBITS        {{.*}} 000004
# KEEP-NEXT:  [ 4] .data.rel.ro.unlikely    PROGBITS        {{.*}} 000002
# KEEP-NEXT:  [ 5] .bss.rel.ro              NOBITS          {{.*}} 000008
# KEEP-NEXT:  [ 6] .relro_padding           NOBITS          {{.*}} 000e20
# KEEP-NEXT:  [ 7] .rodata                  PROGBITS        {{.*}} 000002
# KEEP-NEXT:  [ 8] .rodata.hot              PROGBITS        {{.*}} 000002
# KEEP-NEXT:  [ 9] .rodata.unlikely         PROGBITS        {{.*}} 000002
# KEEP-NEXT:  [10] .data                    PROGBITS        {{.*}} 000002
# KEEP-NEXT:  [11] .data.hot                PROGBITS        {{.*}} 000001
# KEEP-NEXT:  [12] .data.unlikely           PROGBITS        {{.*}} 000001
# KEEP-NEXT:  [13] .bss                     NOBITS          {{.*}} 000002
# KEEP-NEXT:  [14] .bss.hot                 NOBITS          {{.*}} 000001
# KEEP-NEXT:  [15] .bss.unlikely            NOBITS          {{.*}} 000001

# SCRIPT:      .text
# SCRIPT-NEXT: .bss.rel.ro
# SCRIPT-NEXT: .rodata.i
# SCRIPT-NEXT: .rodata.hot.
# SCRIPT-NEXT: .rodata.unlikely.k
# SCRIPT-NEXT: .rodata.split.l
# SCRIPT-NEXT: .rodata.cst32.hot.
# SCRIPT-NEXT: .rodata.str1.1.unlikely.
# SCRIPT-NEXT: .data.m
# SCRIPT-NEXT: .data.hot.n
# SCRIPT-NEXT: .data.unlikely.o
# SCRIPT-NEXT: .data.split.p
# SCRIPT-NEXT: .data.rel.ro.q
# SCRIPT-NEXT: .data.rel.ro.hot.r
# SCRIPT-NEXT: .data.rel.ro.unlikely.s
# SCRIPT-NEXT: .data.rel.ro.split.t
# SCRIPT-NEXT: .bss.a
# SCRIPT-NEXT: .bss.hot.b
# SCRIPT-NEXT: .bss.unlikely.c
# SCRIPT-NEXT: .bss.split.d

#--- x1.lds
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

#--- b.s
.globl _start
_start:
  ret

.section .bss.a,"aw"
  .byte 0
.section .bss.hot.b,"aw"
  .byte 0
.section .bss.unlikely.c,"aw"
  .byte 0
.section .bss.split.d,"aw"
  .byte 0

.section .bss.rel.ro, "aw"
  .space 2
.section .bss.rel.ro, "aw"
  .space 2
.section .bss.rel.ro, "aw"
  .space 2
.section .bss.rel.ro, "aw"
  .space 2

.section .rodata.i,"aw"
  .byte 1
.section .rodata.hot.,"aw"
  .byte 2
.section .rodata.unlikely.k,"aw"
  .byte 3
.section .rodata.split.l,"aw"
  .byte 4
.section .rodata.cst32.hot.,"aw"
  .byte 5
.section .rodata.str1.1.unlikely.,"aw"
  .byte 6

.section .data.m,"aw"
  .byte 5
.section .data.hot.n,"aw"
  .byte 6
.section .data.unlikely.o,"aw"
  .byte 7
.section .data.split.p,"aw"
  .byte 8

.section .data.rel.ro.q,"aw"
  .quad 0 
.section .data.rel.ro.hot.r,"aw"
  .long 255
.section .data.rel.ro.unlikely.s,"aw"
  .word 1
.section .data.rel.ro.split.t,"aw"
  .byte 0

#--- x2.lds
SECTIONS {}
