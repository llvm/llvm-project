# REQUIRES: x86
## Test merging of identical CIEs and the order of CIEs and FDEs in .eh_frame.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: ld.lld a.o a.o -shared --emit-relocs -o a.so
# RUN: llvm-readelf -S -r a.so | FileCheck %s --check-prefixes=CHECK,RELOC
# RUN: llvm-dwarfdump --eh-frame a.so | FileCheck %s --check-prefix=EH

## Also show that the merging happens when going via a -r link.
# RUN: ld.lld -r a.o a.o -o a.ro
# RUN: ld.lld a.ro -o a2.so -shared
# RUN: llvm-readelf -S -r a2.so | FileCheck %s

## i386 uses REL, so the addends are read from the section contents.
# RUN: llvm-mc -filetype=obj -triple=i686 a.s -o a32.o
# RUN: ld.lld a32.o a32.o -shared -o a32.so

# CHECK:       Name      Type     Address              Off      Size   ES Flg Lk Inf Al
# CHECK:      .eh_frame  PROGBITS [[#%x,]]             [[#%x,]] 000058 00   A  0   0  8
# CHECK:      foo        PROGBITS {{0*}}[[#%x,FOO:]]   [[#%x,]] 000002 00  AX  0   0  1
# CHECK-NEXT: bar        PROGBITS {{0*}}[[#%x,FOO+2]]  [[#%x,]] 000002 00  AX  0   0  1

# RELOC:        Offset             Info     Type          Symbol's Value  Symbol's Name + Addend
# RELOC-NEXT: {{0*}}[[#%x,OFF:]]   [[#%x,]] R_X86_64_PC32 [[#%x,]]        foo + 0
# RELOC-NEXT: {{0*}}[[#%x,OFF+20]] [[#%x,]] R_X86_64_PC32 [[#%x,]]        bar + 0
# RELOC-NEXT: {{0*}}[[#OFF+40]]    [[#%x,]] R_X86_64_PC32 [[#%x,]]        foo + 1
# RELOC-NEXT: {{0*}}[[#%x,OFF-24]] [[#%x,]] R_X86_64_NONE 0{{$}}

# EH:          Format:                DWARF32
# EH:        00000018 00000010 0000001c FDE cie=00000000 pc={{0*}}[[#%x,FOO:]]...
# EH-SAME:   {{0*}}[[#%x,FOO+1]]
# EH-COUNT-3:  DW_CFA_nop:
# EH-EMPTY:  
# EH:        0000002c 00000010 00000030 FDE cie=00000000 pc={{0*}}[[#%x,FOO+2]]...{{0*}}[[#%x,FOO+4]]
# EH-COUNT-3:  DW_CFA_nop:
# EH-EMPTY:
# EH:        00000040 00000010 00000044 FDE cie=00000000 pc={{0*}}[[#%x,FOO+1]]...{{0*}}[[#%x,FOO+2]]
# EH-COUNT-3:  DW_CFA_nop:
# EH-EMPTY:
# EH-NEXT:     0x[[#%x,]]: CFA=RSP+8: RIP=[CFA-8]
# EH-EMPTY:
# EH-NEXT:   00000054 ZERO terminator

## Merged CIEs precede all FDEs, which keep input order regardless of the merged
## CIE they refer to.
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld b.o -Ttext=0x201000 -o order
# RUN: llvm-nm order | FileCheck %s --check-prefix=SYM
# RUN: llvm-dwarfdump --eh-frame order | FileCheck %s --check-prefix=ORDER

## Symbols in a dead duplicate CIE are placed at the start of .eh_frame.
# SYM:      [[#%x,BASE:]] r cieA
# SYM-NEXT: {{0*}}[[#%x,BASE]] r cieA2
# SYM-NEXT: {{0*}}[[#%x,BASE]] r cieA3
# SYM-NEXT: {{0*}}[[#%x,BASE+0x10]] r cieB
# SYM-NEXT: {{0*}}[[#%x,BASE]] r cieB2

# ORDER:      00000000 0000000c 00000000 CIE
# ORDER:      00000010 0000000c 00000000 CIE
# ORDER:      FDE cie=00000000 pc=00201000...
# ORDER:      FDE cie=00000010 pc=00201001...
# ORDER:      FDE cie=00000000 pc=00201002...
# ORDER:      FDE cie=00000000 pc=00201003...
# ORDER:      FDE cie=00000010 pc=00201005...
# ORDER:      FDE cie=00000000 pc=00201004...
# ORDER-NOT:  {{CIE|FDE}}

## CIEs with different personalities are not merged.
# RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
# RUN: ld.lld --eh-frame-hdr -shared c.o -o c.so
# RUN: llvm-dwarfdump --eh-frame c.so | FileCheck %s --check-prefix=PERS

# PERS:      00000000 00000014 00000000 CIE
# PERS:        Augmentation: "zR"
# PERS:      00000018 00000018 00000000 CIE
# PERS:        Augmentation: "zPR"
# PERS:        Personality Address: [[#%.16x,P0:]]
# PERS:      00000034 00000018 00000000 CIE
# PERS:        Augmentation: "zPR"
# PERS:        Personality Address: [[#%.16x,P0+1]]
# PERS:      00000050 00000010 00000054 FDE cie=00000000
# PERS:      00000064 00000010 00000050 FDE cie=00000018
# PERS:      00000078 00000014 00000048 FDE cie=00000034

#--- a.s
        .section	foo,"ax",@progbits
	.cfi_startproc
        nop
	.cfi_endproc

        .section	bar,"axG",@progbits,foo,comdat
        .cfi_startproc
        nop
        nop
	.cfi_endproc

#--- b.s
.macro cie name, caf
\name: .long 12, 0
  .byte 1, 0, \caf, 0x78, 16, 0, 0, 0
.endm
.macro fde cie, fn
  .long 20
  .long . - \cie
  .quad \fn, 1
.endm

.globl _start
.text
_start:
p: nop
q: nop
x: nop
y: nop
v: nop
w: nop

## CIE A, FDE p->A, CIE B, FDE q->B, CIE A2 (duplicate of A), FDE x->A2, FDE y->A.
.section .eh_frame,"a",@unwind
cie cieA, 1
fde cieA, p
cie cieB, 2
fde cieB, q
cie cieA2, 1
fde cieA2, x
fde cieA, y

## In another .eh_frame section, the FDE of w precedes that of v but goes to the later CIE.
.section .eh_frame,"a",@unwind,unique,1
cie cieB2, 2
fde cieB2, w
cie cieA3, 1
fde cieA3, v

#--- c.s
foo0:
.cfi_startproc
.cfi_personality 0x9b, personality0
  ret
.cfi_endproc

## This CIE cannot be merged into the previous one because the Personality is different.
foo1:
.cfi_startproc
.cfi_personality 0x9b, personality1
  ret
.cfi_endproc

bar:
.cfi_startproc
  ret
.cfi_endproc

.globl personality0, personality1
.hidden personality0, personality1
personality0:
  ret

personality1:
  ret
