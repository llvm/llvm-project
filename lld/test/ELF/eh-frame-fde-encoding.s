# REQUIRES: x86
## Test that various DW_EH_PE_* encodings in CIE are accepted.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 absptr.s -o absptr.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 sdata2.s -o sdata2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 sdata4.s -o sdata4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 sdata8.s -o sdata8.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 signed.s -o signed.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 udata2.s -o udata2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 udata4.s -o udata4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 udata8.s -o udata8.o

# RUN: ld.lld --eh-frame-hdr sdata2.o -o /dev/null
# RUN: ld.lld --eh-frame-hdr sdata8.o -o /dev/null
# RUN: ld.lld --eh-frame-hdr signed.o -o /dev/null

# RUN: ld.lld --eh-frame-hdr --image-base=0 -Ttext=0x1000 absptr.o -o absptr
# RUN: ld.lld --eh-frame-hdr --image-base=0 -Ttext=0x1000 udata2.o -o udata2
# RUN: ld.lld --eh-frame-hdr --image-base=0 -Ttext=0x2000 sdata4.o -o sdata4
# RUN: ld.lld --eh-frame-hdr --image-base=0 -Ttext=0x2000 udata4.o -o udata4

## absptr/udata2: Also verify .eh_frame content to test relocation with addend.
## .eh_frame_hdr initial_location: foo(0x1000)+0x234 - .eh_frame_hdr(0x2004) = 0xfffff230
# RUN: llvm-readobj -x .eh_frame_hdr -x .eh_frame absptr | FileCheck %s --check-prefix=ABSPTR
# ABSPTR:      Hex dump of section '.eh_frame_hdr':
# ABSPTR-NEXT: 0x00002004 011b033b 10000000 01000000 30f2ffff
# ABSPTR-NEXT: 0x00002014 24000000
# ABSPTR:      Hex dump of section '.eh_frame':
# ABSPTR-NEXT: 0x00002018 0c000000 00000000 01520001 010100ff
# ABSPTR-NEXT: 0x00002028 0c000000 14000000 34120000 00000000
##                        CIE offset--^     ^-- PC begin = 0x1234 (foo + 0x234)

# RUN: llvm-readobj -x .eh_frame_hdr -x .eh_frame udata2 | FileCheck %s --check-prefix=UDATA2
# UDATA2:      Hex dump of section '.eh_frame_hdr':
# UDATA2-NEXT: 0x00002004 011b033b 10000000 01000000 30f2ffff
# UDATA2-NEXT: 0x00002014 26000000
# UDATA2:      Hex dump of section '.eh_frame':
# UDATA2-NEXT: 0x00002018 0e000000 00000000 01525300 01010102
# UDATA2-NEXT: 0x00002028 ff000600 00001600 00003412
##                        CIE offset--^     ^-- PC begin = 0x1234 (foo + 0x234)

# RUN: llvm-readelf -x .eh_frame_hdr sdata4 udata4 | FileCheck %s --check-prefix=HDR4
# HDR4:      0x00003004 011b033b 10000000 01000000 fcefffff
# HDR4-NEXT: 0x00003014 24000000
# HDR4:      0x00003004 011b033b 10000000 01000000 fcefffff
# HDR4-NEXT: 0x00003014 24000000

#--- absptr.s
## DW_EH_PE_absptr (0x00) with FDE for verification
.text
.globl foo
foo:
  nop

.section .eh_frame,"a",@unwind
  .long 12          # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x52        # Augmentation string: 'R','\0'
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment
  .byte 0x01        # Return address register
  .byte 0x00        # DW_EH_PE_absptr
  .byte 0xFF

  .long 12          # Size
  .long 0x14        # CIE offset
  .quad foo + 0x234 # PC begin

#--- sdata2.s
## DW_EH_PE_sdata2 (0x0A)
.section .eh_frame,"a",@unwind
  .long 0x0E        # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x50        # Augmentation string: 'P','\0'
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment (LEB128)
  .byte 0x01        # Return address register (LEB128)
  .byte 0x0A        # DW_EH_PE_sdata2
  .short 0xFFFF
  .byte 0xFF

#--- sdata4.s
## DW_EH_PE_sdata4 (0x0B) with FDE for verification
.text
.globl foo
foo:
  nop

.section .eh_frame,"a",@unwind
  .long 12          # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x52        # Augmentation string: 'R','\0'
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment
  .byte 0x01        # Return address register
  .byte 0x0B        # DW_EH_PE_sdata4
  .byte 0xFF

  .long 12          # Size
  .long 0x14        # CIE offset
  .long foo         # PC begin
  .long 1           # PC range

#--- sdata8.s
## DW_EH_PE_sdata8 (0x0C)
.section .eh_frame,"a",@unwind
  .long 0x14        # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x50        # Augmentation string: 'P','\0'
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment (LEB128)
  .byte 0x01        # Return address register (LEB128)
  .byte 0x0C        # DW_EH_PE_sdata8
  .quad 0xFFFFFFFFFFFFFFFF
  .byte 0xFF

#--- signed.s
## DW_EH_PE_signed (0x08)
.section .eh_frame,"a",@unwind
  .long 0x14        # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x50        # Augmentation string: 'P','\0'
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment (LEB128)
  .byte 0x01        # Return address register (LEB128)
  .byte 0x08        # DW_EH_PE_signed
  .quad 0xFFFFFFFFFFFFFFFF
  .byte 0xFF

#--- udata2.s
## DW_EH_PE_udata2 (0x02) with FDE for verification
.text
.globl foo
foo:
  nop

.section .eh_frame,"a",@unwind
  .long 14          # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x52        # Augmentation string: 'R','S','\0'
  .byte 0x53
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment
  .byte 0x01        # Return address register
  .byte 0x02        # DW_EH_PE_udata2
  .byte 0xFF
  .byte 0x00

  .long 6           # Size
  .long 0x16        # CIE offset
  .short foo + 0x234  # PC begin

#--- udata4.s
## DW_EH_PE_udata4 (0x03) with FDE for verification
.text
.globl foo
foo:
  nop

.section .eh_frame,"a",@unwind
  .long 12          # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x52        # Augmentation string: 'R','\0'
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment
  .byte 0x01        # Return address register
  .byte 0x03        # DW_EH_PE_udata4
  .byte 0xFF

  .long 12          # Size
  .long 0x14        # CIE offset
  .long foo         # PC begin
  .long 1           # PC range

#--- udata8.s
## DW_EH_PE_udata8 (0x04)
.section .eh_frame,"a",@unwind
  .long 0x14        # Size
  .long 0x00        # ID (CIE)
  .byte 0x01        # Version
  .byte 0x50        # Augmentation string: 'P','\0'
  .byte 0x00
  .byte 0x01        # Code alignment
  .byte 0x01        # Data alignment (LEB128)
  .byte 0x01        # Return address register (LEB128)
  .byte 0x04        # DW_EH_PE_udata8
  .quad 0xFFFFFFFFFFFFFFFF
  .byte 0xFF
