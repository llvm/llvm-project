# REQUIRES: x86

## Check that we correctly skip FDEs with zero-size PC ranges as
## they may hide another non-trivial FDE at the same address.
## Here, f1 has the same address as f2, but we must use f2's
## FDE in the header.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld --eh-frame-hdr %t.o -o %t
# RUN: llvm-readelf -S -x .eh_frame_hdr %t | FileCheck %s

# CHECK:      Hex dump of section '.eh_frame_hdr':
# CHECK-NEXT: 011b033b 1c000000 01000000 {{[0-9a-f]+}}
# CHECK-NEXT: 4c000000
##            ^ offset of second FDE

.text
.globl f1, f2

f1:
  .cfi_startproc
  .cfi_endproc

f2:
  .cfi_startproc
  ret
  .cfi_endproc
