# REQUIRES: x86
## .eh_frame_hdr's size feeds back into its own contents: it shifts downstream
## addresses, which decides whether two FDEs share an initial location and get
## deduplicated, which changes the FDE count and therefore the size again. Any
## output section boundary whose alignment is comparable to the size delta can
## close the loop.
##
## fz is zero-length at the end of .text.a. With a 36-byte .eh_frame_hdr, .text.a
## ends 16-byte aligned, and fz and fb share an address. Deduplicating their
## FDEs gives a 28-byte header, which shifts # .text.b by 8 and separates
## them again. Clamping the size converges.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: ld.lld -T lds --eh-frame-hdr a.o -o a
# RUN: llvm-readelf -S -x .eh_frame_hdr a | FileCheck %s

## The header keeps room for 3 FDEs while fde_count is 2. The trailing entry is
## unwritten; consumers use fde_count.
# CHECK:      .eh_frame_hdr PROGBITS 0000000000010000 001000 000024
# CHECK:      Hex dump of section '.eh_frame_hdr':
# CHECK-NEXT: 0x00010000 011b033b {{[0-9a-f]+}} 02000000 {{[0-9a-f]+}}
# CHECK-NEXT: 0x00010010 {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} 00000000
# CHECK-NEXT: 0x00010020 00000000

#--- lds
SECTIONS {
  . = 0x10000;
  .eh_frame_hdr : { *(.eh_frame_hdr) }
  .eh_frame : { *(.eh_frame) }
  .text.a : { *(.text.a) }
  .text.b : ALIGN(16) { *(.text.b) }
}

#--- a.s
.globl _start
.section .text.a, "ax", @progbits
_start:
  .cfi_startproc
## Sized so that .text.a ends 16-byte aligned in the 36-byte layout.
  .space 12, 0x90
  .cfi_endproc
fz:
  .cfi_startproc
  .cfi_endproc

.section .text.b, "ax", @progbits
fb:
  .cfi_startproc
  nop
  .cfi_endproc
