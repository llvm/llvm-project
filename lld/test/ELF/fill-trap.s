# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bss.s -o bss.o

## -z noseparate-code is the default: text segment is not tail padded.
# RUN: ld.lld a.o -o out
# RUN: llvm-readobj -l out | FileCheck %s --check-prefixes=CHECK,NOPAD
# RUN: ld.lld a.o -z noseparate-code -z common-page-size=512 -o out
# RUN: llvm-readobj -l out | FileCheck %s --check-prefixes=CHECK,NOPAD

## -z separate-code pads the tail of text segment with traps.
## Make common-page-size smaller than max-page-size.
## Check that we use max-page-size instead of common-page-size for padding.
# RUN: ld.lld a.o -z separate-code -z common-page-size=512 -o out
# RUN: llvm-readobj -l out | FileCheck %s --check-prefixes=CHECK,PAD
# RUN: od -Ax -x -N16 -j0x1ff0 out | FileCheck %s --check-prefix=FILL

## -z separate-loadable-segments pads all segments, including the text segment.
# RUN: ld.lld a.o -z separate-loadable-segments -z common-page-size=512 -o out
# RUN: llvm-readobj -l out | FileCheck %s --check-prefixes=CHECK,PAD
# RUN: od -Ax -x -N16 -j0x1ff0 out | FileCheck %s --check-prefix=FILL

# RUN: ld.lld a.o -z separate-code -z noseparate-code -z common-page-size=512 -o out
# RUN: llvm-readobj -l out | FileCheck %s --check-prefixes=CHECK,NOPAD

# CHECK: ProgramHeader {
# CHECK:   Type: PT_LOAD
# PAD:     Offset: 0x1000
# NOPAD:   Offset: 0x120
# CHECK-NEXT:   VirtualAddress:
# CHECK-NEXT:   PhysicalAddress:
# PAD-NEXT:     FileSize: 4096
# NOPAD-NEXT:   FileSize: 1
# CHECK-NEXT:   MemSize:
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     PF_R
# CHECK-NEXT:     PF_X
# CHECK-NEXT:   ]

## Check that executable page is filled with traps at its end.
# FILL: 001ff0 cccc cccc cccc cccc cccc cccc cccc cccc

## There is a single RWX segment. Test that p_memsz is not truncated to p_filesz.
# RUN: ld.lld a.o bss.o -z separate-loadable-segments -T rwx.lds -z max-page-size=64 -o rwx
# RUN: llvm-readelf -l rwx | FileCheck %s --check-prefix=RWX

# RWX:       Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# RWX-NEXT:  LOAD           0x000080 0x0000000000000000 0x0000000000000000 0x000040 0x000404 RWE 0x40

# RUN: ld.lld a.o bss.o -z separate-loadable-segments --omagic -o omagic
# RUN: llvm-readelf -l omagic | FileCheck %s --check-prefix=OMAGIC

# OMAGIC:     Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# OMAGIC-NEXT:LOAD           0x0000b0 0x00000000002000b0 0x00000000002000b0 0x000004 0x000404 RWE 0x4

## Test that gaps between sections within an executable segment are filled with traps.
# RUN: llvm-mc -filetype=obj -triple=x86_64 gap.s -o gap.o
# RUN: ld.lld gap.o -z separate-code -z max-page-size=0x1000 -o gap.out
## .text is at offset 0x1000, .text2 is aligned to 16 bytes at 0x1010.
## The gap between them should be filled with 0xcc.
# RUN: od -Ax -t x1 -v -N32 -j0x1000 gap.out | FileCheck %s --check-prefix=GAP
# GAP:      001000 90 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
# GAP-NEXT: 001010 90 cc

## Test multiple gaps with various alignments.
## Sections: .text (1 byte) -> .text2 (align 8) -> .text3 (align 32) -> .text4 (align 128)
# RUN: llvm-mc -filetype=obj -triple=x86_64 multi-gap.s -o multi-gap.o
# RUN: ld.lld multi-gap.o -z separate-code -z max-page-size=0x1000 -o multi-gap.out
# RUN: od -Ax -t x1 -v -N144 -j0x1000 multi-gap.out | FileCheck %s --check-prefix=MGAP
## .text at 0x1000, .text2 at 0x1008 (aligned to 8)
# MGAP:      001000 90 cc cc cc cc cc cc cc 90 cc cc cc cc cc cc cc
## gap from 0x1009 to 0x1020, .text3 at 0x1020 (aligned to 32)
# MGAP-NEXT: 001010 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
# MGAP-NEXT: 001020 90 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
## gap from 0x1021 to 0x1080
# MGAP-NEXT: 001030 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
# MGAP-NEXT: 001040 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
# MGAP-NEXT: 001050 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
# MGAP-NEXT: 001060 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
# MGAP-NEXT: 001070 cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc
## .text4 at 0x1080 (aligned to 128)
# MGAP-NEXT: 001080 90 cc

#--- a.s
.globl _start
_start:
  nop

#--- bss.s
.bss
.space 1024

#--- gap.s
.globl _start
.section .text,"ax"
_start:
  nop

.section .text2,"ax"
.p2align 4
  nop

#--- multi-gap.s
.globl _start
.section .text,"ax"
_start:
  nop

.section .text2,"ax"
.p2align 3
  nop

.section .text3,"ax"
.p2align 5
  nop

.section .text4,"ax"
.p2align 7
  nop

#--- rwx.lds
PHDRS { all PT_LOAD; }
SECTIONS {
  .text : {*(.text*)} :all
  .bss : {*(.bss*)} :all
}
