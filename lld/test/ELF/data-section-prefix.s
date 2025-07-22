# REQUIRES: x86
## -z keep-data-section-prefix separates static data sections with prefix
## .<section>.hot, .<section>.unlikely in the absence of a SECTIONS command.

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld a.o -o out1
# RUN: llvm-readelf -S out1 | FileCheck --check-prefix=BASIC %s
# RUN: ld.lld -z nokeep-text-section-prefix a.o -o out2
# RUN: cmp out1 out2

# RUN: ld.lld -z keep-data-section-prefix a.o -o out3
# RUN: llvm-readelf -S out3 | FileCheck --check-prefix=KEEP %s

## With a SECTIONS command, orphan sections are created verbatim.
## No grouping is performed for them.
# RUN: ld.lld -T x.lds -z keep-data-section-prefix a.o -o out4
# RUN: llvm-readelf -S out4 | FileCheck --check-prefix=SCRIPT %s

# BASIC:         [Nr] Name              Type            Address          Off    Size              
# BASIC:         [ 1] .text       
# BASIC-NEXT:    [ 2] .data.rel.ro      PROGBITS        00000000002021c9 0001c9 00000f
# BASIC-NEXT:    [ 3] .bss.rel.ro       NOBITS          00000000002021d8 0001d8 000008
# BASIC-NEXT:    [ 4] .relro_padding    NOBITS          00000000002021e0 0001d8 000e20
# BASIC-NEXT:    [ 5] .rodata           PROGBITS        00000000002031e0 0001e0 000006
# BASIC-NEXT:    [ 6] .data             PROGBITS        00000000002031e6 0001e6 000004
# BASIC-NEXT:    [ 7] .bss              NOBITS          00000000002031ea 0001ea 000004

# KEEP:       [Nr]  Name                    Type            Address          Off    Size
# KEEP:       [ 1] .text
# KEEP-NEXT:  [ 2] .data.rel.ro             PROGBITS        00000000002021c9 0001c9 000009
# KEEP-NEXT:  [ 3] .data.rel.ro.hot         PROGBITS        00000000002021d2 0001d2 000004
# KEEP-NEXT:  [ 4] .data.rel.ro.unlikely    PROGBITS        00000000002021d6 0001d6 000002
# KEEP-NEXT:  [ 5] .bss.rel.ro              NOBITS          00000000002021d8 0001d8 000008
# KEEP-NEXT:  [ 6] .relro_padding           NOBITS          00000000002021e0 0001d8 000e20
# KEEP-NEXT:  [ 7] .rodata                  PROGBITS        00000000002031e0 0001e0 000002
# KEEP-NEXT:  [ 8] .rodata.hot              PROGBITS        00000000002031e2 0001e2 000002
# KEEP-NEXT:  [ 9] .rodata.unlikely         PROGBITS        00000000002031e4 0001e4 000002
# KEEP-NEXT:  [10] .data                    PROGBITS        00000000002031e6 0001e6 000002
# KEEP-NEXT:  [11] .data.hot                PROGBITS        00000000002031e8 0001e8 000001
# KEEP-NEXT:  [12] .data.unlikely           PROGBITS        00000000002031e9 0001e9 000001
# KEEP-NEXT:  [13] .bss                     NOBITS          00000000002031ea 0001ea 000002
# KEEP-NEXT:  [14] .bss.hot                 NOBITS          00000000002031ec 0001ea 000001
# KEEP-NEXT:  [15] .bss.unlikely            NOBITS          00000000002031ed 0001ea 000001

# SCRIPT:      .text
# SCRIPT-NEXT: .rodata.i
# SCRIPT-NEXT: .rodata.hot.j
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
# SCRIPT-NEXT: .bss.rel.ro.e
# SCRIPT-NEXT: .bss.rel.ro.hot.f
# SCRIPT-NEXT: .bss.rel.ro.unlikely.g
# SCRIPT-NEXT: .bss.rel.ro.split.h

#--- a.s
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
.section .rodata.hot.j,"aw"
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

#--- x.lds
SECTIONS {}
