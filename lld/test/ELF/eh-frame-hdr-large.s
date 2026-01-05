# REQUIRES: x86

# Test that .eh_frame_hdr uses 64-bit encodings (sdata8) when offsets exceed
# 32-bit range. This can happen with very large binaries even with the large code model.
# https://github.com/llvm/llvm-project/issues/172777

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s --large-code-model -o %t.o

## Test 1: Place .text at a high address (>4GB from .eh_frame_hdr) using linker script.
## The eh_frame_hdr should use sdata8 encoding (0x0C = DW_EH_PE_sdata8).
# RUN: echo "SECTIONS { \
# RUN:   .eh_frame_hdr : { *(.eh_frame_hdr) } \
# RUN:   .eh_frame : { *(.eh_frame) } \
# RUN:   .text 0x100000000 : { *(.text) } \
# RUN: }" > %t-large.lds
# RUN: ld.lld --eh-frame-hdr --script %t-large.lds %t.o -o %t-large
# RUN: llvm-readobj -S --section-data %t-large | FileCheck %s

# CHECK:      Section {
# CHECK:        Name: .eh_frame_hdr
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK:        Size: 32
#                     ^^ The size should be: 4 + 8 (large eh_frame_ptr) + 4 (small fde_count) + 16 (one large entry)
#                        = 4 + 8 + 4 + 16 = 32
.text
.global _start
_start:
 .cfi_startproc
 nop
 .cfi_endproc
