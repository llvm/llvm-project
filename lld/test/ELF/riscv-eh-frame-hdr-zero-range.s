# REQUIRES: riscv
## A zero-sized function shares its address with the next function. Its zero-range
## FDE is discarded in .eh_frame_hdr so as not to displace the next function's FDE.
## On RISC-V with linker relaxation, pc_range has relocations (ADD32/SUB32) leaving
## un-relocated zeroes in the object file. Check that non-zero functions are
## retained in .eh_frame_hdr, while truly zero-range functions are discarded.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c %s -o %t.o
# RUN: ld.lld --eh-frame-hdr --defsym=foo=0x20000 %t.o -o %t
# RUN: llvm-readelf --unwind %t | FileCheck %s

# CHECK:      fde_count: 1
# CHECK-NEXT: entry 0 {
# CHECK-NEXT:   initial_location: 0x[[#%x,START:]]
# CHECK-NEXT:   address: 0x[[#%x,FDE:]]
# CHECK-NEXT: }

# CHECK:      ] FDE length=
# CHECK-NEXT:   initial_location: 0x[[#START]]
# CHECK-NEXT:   address_range: 0x0
# CHECK:      [0x[[#FDE]]] FDE length=
# CHECK-NEXT:   initial_location: 0x[[#START]]
# CHECK-NEXT:   address_range: 0xa
# CHECK:      ] FDE length=
# CHECK-NEXT:   initial_location: 0x[[#START+0xa]]
# CHECK-NEXT:   address_range: 0x0

f1:
  .cfi_startproc
  .cfi_endproc

.globl _start
_start:
  .cfi_startproc
  call foo
  ret
  .cfi_endproc

f2:
  .cfi_startproc
  .cfi_endproc
