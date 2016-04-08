# Check reading PC values of FDEs and writing lookup table in the .eh_frame_hdr
# if CIE augmentation string has 'L' token and PC values are encoded using
# absolute (not relative) format.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld --eh-frame-hdr %t.o -o %t
# RUN: llvm-objdump -s -dwarf=frames %t | FileCheck %s

# REQUIRES: mips

# CHECK:      Contents of section .eh_frame_hdr:
# CHECK-NEXT:  10158 011b033b ffffffcc 00000001 0000fea8
#                                               ^-- 0x20000 - 0x10158
#                                                   .text   - .eh_frame_hdr
# CHECK-NEXT:  10168 ffffffe8
# CHECK-NEXT: Contents of section .text:
# CHECK-NEXT:  20000 00000000

# CHECK: Augmentation:          "zLR"
# CHECK: Augmentation data:     00 0B
#                                  ^-- DW_EH_PE_udata4 | DW_EH_PE_signed

	.text
  .globl __start
__start:
	.cfi_startproc
  .cfi_lsda 0, _ex
  nop
	.cfi_endproc

  .data
_ex:
  .word 0
