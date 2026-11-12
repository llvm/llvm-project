# REQUIRES: x86
## f1 is empty and shares its address with _start. Its zero-range FDE is
## omitted from .eh_frame_hdr so as not to displace _start's FDE.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --eh-frame-hdr %t.o -o %t
# RUN: llvm-readelf --unwind %t | FileCheck %s

# CHECK:      fde_count: 1
# CHECK-NEXT: entry 0 {
# CHECK-NEXT:   initial_location: 0x[[#%x,START:]]
# CHECK-NEXT:   address: 0x[[#%x,FDE:]]
# CHECK:      [0x[[#FDE]]] FDE length=
# CHECK-NEXT:   initial_location: 0x[[#START]]
# CHECK-NEXT:   address_range: 0x1

f1:
  .cfi_startproc
  .cfi_endproc

.globl _start
_start:
  .cfi_startproc
  ret
  .cfi_endproc
