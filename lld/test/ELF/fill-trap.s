# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o

## -z noseparate-code is the default: text segment is not tail padded.
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj -l %t | FileCheck %s --check-prefixes=CHECK,NOPAD
# RUN: ld.lld %t.o -z noseparate-code -o %t
# RUN: llvm-readobj -l %t | FileCheck %s --check-prefixes=CHECK,NOPAD

## -z separate-code pads the tail of text segment with traps.
# RUN: ld.lld %t.o -z separate-code -o %t
# RUN: llvm-readobj -l %t | FileCheck %s --check-prefixes=CHECK,PAD
# RUN: od -Ax -x -N16 -j0x1ff0 %t | FileCheck %s --check-prefix=FILL

## -z separate-loadable-segments pads all segments, including the text segment.
# RUN: ld.lld %t.o -z separate-loadable-segments -o %t
# RUN: llvm-readobj -l %t | FileCheck %s --check-prefixes=CHECK,PAD
# RUN: od -Ax -x -N16 -j0x1ff0 %t | FileCheck %s --check-prefix=FILL

# RUN: ld.lld %t.o -z separate-code -z noseparate-code -o %t
# RUN: llvm-readobj -l %t | FileCheck %s --check-prefixes=CHECK,NOPAD

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

.globl _start
_start:
  nop
