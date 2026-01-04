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

#--- a.s
.globl _start
_start:
  nop

#--- bss.s
.bss
.space 1024

#--- rwx.lds
PHDRS { all PT_LOAD; }
SECTIONS {
  .text : {*(.text*)} :all
  .bss : {*(.bss*)} :all
}
