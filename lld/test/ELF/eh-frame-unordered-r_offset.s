# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: cp %t/a.o %t/b.o
# RUN: ld.lld -r -T %t/lds %t/a.o %t/b.o -o %t/c.o
# RUN: llvm-readelf -r %t/c.o | FileCheck %s --check-prefix=REL

## If we swap two input .eh_frame, the r_offset values in relocations will be
## unordered.
# REL:          Offset
# REL-NEXT: 0000000000000050
# REL-NEXT: 0000000000000020

## Test we can handle the rare case.
# RUN: ld.lld %t/c.o -o %t/c
# RUN: llvm-dwarfdump --eh-frame %t/c | FileCheck %s

# CHECK: 00000000 00000014 00000000 CIE
# CHECK: 00000018 00000014 0000001c FDE cie=00000000
# CHECK: 00000030 00000014 00000034 FDE cie=00000000

#--- a.s
.cfi_startproc
nop
.cfi_endproc

#--- lds
SECTIONS {
  .eh_frame : { *b.o(.eh_frame) *a.o(.eh_frame) }
}
