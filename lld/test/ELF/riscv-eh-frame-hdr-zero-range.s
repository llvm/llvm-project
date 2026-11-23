# REQUIRES: riscv
## With linker relaxation, GNU as computes pc_range with an ADD/SUB relocation
## pair, even for an empty function. f1 is empty and shares its address with
## _start. Its zero-range FDE is omitted from .eh_frame_hdr.

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: ld.lld --eh-frame-hdr %t.o -o %t
# RUN: llvm-readelf --unwind %t | FileCheck %s

# CHECK:      fde_count: 1
# CHECK-NEXT: entry 0 {
# CHECK-NEXT:   initial_location: 0x[[#%x,START:]]
# CHECK-NEXT:   address: 0x[[#%x,FDE:]]
# CHECK:      [0x[[#FDE]]] FDE length=
# CHECK-NEXT:   initial_location: 0x[[#START]]
# CHECK-NEXT:   address_range: 0x4

.globl _start
f1:
_start:
  ret
.Lend:

.section .eh_frame,"a",@progbits
  .long 16, 0                # CIE length, CIE ID
  .byte 1                    # Version
  .asciz "zR"
  .byte 1, 0x7c, 1, 1, 0x1b  # Code/data alignment, RA register, augmentation data
  .balign 4

  .long 16, 24               # FDE length, CIE pointer
  .reloc ., R_RISCV_32_PCREL, f1
  .reloc .+4, R_RISCV_ADD32, f1
  .reloc .+4, R_RISCV_SUB32, f1
  .long 0, 0, 0              # pc_begin, pc_range, augmentation data length

  .long 16, 44
  .reloc ., R_RISCV_32_PCREL, _start
  .reloc .+4, R_RISCV_ADD32, .Lend
  .reloc .+4, R_RISCV_SUB32, _start
  .long 0, 0, 0
