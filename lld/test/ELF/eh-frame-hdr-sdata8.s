# REQUIRES: x86 && llvm-64-bits

## Test that .eh_frame_hdr uses DW_EH_PE_sdata8 instead of DW_EH_PE_sdata4 when
## eh_frame_ptr or a table entry exceeds the 32-bit range.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s --large-code-model -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad.s -o bad.o

## Case 1: .text at high address - pcRel exceeds 32-bit range.
# RUN: ld.lld --eh-frame-hdr -T 1.lds a.o -o out1
# RUN: llvm-objdump -s -j .eh_frame_hdr out1 | FileCheck %s --check-prefix=CHECK1

## Case 2: .eh_frame at high address - eh_frame_ptr exceeds 32-bit range.
# RUN: ld.lld --eh-frame-hdr -T 2.lds a.o -o out2
# RUN: llvm-objdump -s -j .eh_frame_hdr out2 | FileCheck %s --check-prefix=CHECK2

## Case 3: .eh_frame_hdr and .relr.dyn sizes are coupled, requiring multiple iterations
## to stabilize.
# RUN: ld.lld -pie --eh-frame-hdr -z pack-relative-relocs -T 3.lds a.o -o out3
# RUN: llvm-objdump -s -j .eh_frame_hdr out3 | FileCheck %s --check-prefix=CHECK3

## Header: version=1, eh_frame_ptr_enc=0x1C (pcrel|sdata8),
##         fde_count_enc=0x03 (udata4), table_enc=0x3C (datarel|sdata8)
## Layout: header (4) + eh_frame_ptr (8) + fde_count (4) + table entries (16 each)
## Each table entry: pcRel (8) + fdeVARel (8), relative to .eh_frame_hdr address
# CHECK1:      section .eh_frame_hdr:
# CHECK1-NEXT: 011c033c 2c000000 00000000 02000000
# CHECK1-NEXT: 00100000 01000000 48000000 00000000
# CHECK1-NEXT: 01100000 01000000 68000000 00000000
# CHECK1-EMPTY:
# CHECK2:      section .eh_frame_hdr:
# CHECK2-NEXT: 011c033c f80f0000 01000000 02000000
# CHECK2-NEXT: fcffffff ffffffff 14100000 01000000
# CHECK2-NEXT: fdffffff ffffffff 34100000 01000000
# CHECK2-EMPTY:
# CHECK3:      011c033c 2c000000 00000000 02000000
# CHECK3-NEXT: 00800000 01000000 48000000 00000000

## A corrupted .eh_frame reports exactly one error (not duplicated by the loop).
# RUN: not ld.lld --eh-frame-hdr -T 1.lds a.o bad.o 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# ERR: error: corrupted .eh_frame: unexpected end of CIE

#--- a.s
.text
.global _start
_start:
 .cfi_startproc
 nop
 .cfi_endproc
 .cfi_startproc
 nop
 .cfi_endproc

.data
.balign 8
## Two adjacent relocations use 2 RELR entries (1 address + 1 bitmap).
.dc.a __ehdr_start
.dc.a __ehdr_start

.section .data.1,"aw"
.balign 8
## A RELR bitmap entry can encode up to 63 relocations with word-sized stride.
## If .data.1 is >= 63*8 bytes from end(.data), this relocation cannot reuse
## the previous bitmap entry, requiring a third RELR entry.
.dc.a __ehdr_start

#--- 1.lds
## Use AT() to place output sections with huge addresses in separate PT_LOAD
## segments, avoiding a huge PT_LOAD whose sparse file size would exceed 4GiB.
SECTIONS {
  . = 0x1000;
  .eh_frame_hdr : {}
  .eh_frame : {}
  .text 0x100002000 : AT(0x2000) {}
}

#--- 2.lds
SECTIONS {
  . = 0x1000;
  .text : {}
  .eh_frame_hdr : {}
  .eh_frame 0x100002000 : AT(0x2000) {}
}

#--- 3.lds
SECTIONS {
  ## Test that .eh_frame_hdr and .relr.dyn sizes are coupled, requiring
  ## multiple finalizeAddressDependentContent iterations to converge.
  ##
  ## The padding before .data.1 is set so that switching .eh_frame_hdr from
  ## sdata4 (18 bytes) to sdata8 (48 bytes) pushes .data.1 past the 63*8-byte
  ## RELR bitmap threshold, growing .relr.dyn from 16 to 24 bytes.
  ## The .text address depends on SIZEOF(.relr.dyn), creating the coupling.
  .eh_frame_hdr : {}
  .eh_frame : {}
  .relr.dyn : {}
  .data : { *(.data) . += 63*8-40 + SIZEOF(.eh_frame_hdr); *(.data.*) }
  . = SIZEOF(.relr.dyn) > 16 ? 0x100008000 : 0x3000;
  .text : AT(0x2000) {}
  ASSERT(SIZEOF(.relr.dyn) > 16, ".relr.dyn size should increase from 16 to 24")
}

#--- bad.s
## Malformed CIE: length says 8 bytes but content is truncated.
.section .eh_frame,"a",@unwind
  .long 8       # length
  .long 0       # CIE id
  .byte 1       # version
  .byte 0       # augmentation string (empty)
  ## Missing: code/data alignment, return column, etc.
  .space 2
