# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## Test that we correctly handle the sdata4 DWARF pointer encoding. llvm-mc's
## CFI directives always generate EH frames using the absptr (i.e. system
## pointer size) encoding, but it is possible to hand-roll your own EH frames
## that use the sdata4 encoding. For instance, libffi does this.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 %t/sdata4.s -o %t/sdata4.o
# RUN: %lld -lSystem %t/sdata4.o -o %t/sdata4
# RUN: llvm-objdump --macho --syms --dwarf=frames %t/sdata4 | FileCheck %s

# CHECK: SYMBOL TABLE:
# CHECK: [[#%.16x,MAIN:]] g     F __TEXT,__text _main

# CHECK: .eh_frame contents:
# CHECK: 00000000 00000010 00000000 CIE
# CHECK:   Format:                DWARF32
# CHECK:   Version:               1
# CHECK:   Augmentation:          "zR"
# CHECK:   Code alignment factor: 1
# CHECK:   Data alignment factor: 1
# CHECK:   Return address column: 1
# CHECK:   Augmentation data:     1B
# CHECK:   DW_CFA_def_cfa: reg7 +8
# CHECK:   CFA=reg7+8

# CHECK: 00000014 00000010 00000018 FDE cie=00000000 pc=[[#%x,MAIN]]...[[#%x,MAIN+1]]
# CHECK:   Format:       DWARF32
# CHECK:   DW_CFA_GNU_args_size: +16
# CHECK:   DW_CFA_nop:
# CHECK:   0x[[#%x,MAIN]]: CFA=reg7+8

#--- sdata4.s
.globl  _main
_main:
  retq
LmainEnd:

.balign 4
.section __TEXT,__eh_frame
# Although we don't reference this EhFrame symbol directly, we must have at
# least one non-local symbol in this section, otherwise llvm-mc generates bogus
# subtractor relocations.
EhFrame:
LCieHdr:
  .long LCieEnd - LCieStart
LCieStart:
  .long 0           # CIE ID
  .byte 1           # CIE version
  .ascii "zR\0"
  .byte 1           # Code alignment
  .byte 1           # Data alignment
  .byte 1           # RA column
  .byte 1           # Augmentation size
  .byte 0x1b        # FDE pointer encoding (pcrel | sdata4)
  .byte 0xc, 7, 8   # DW_CFA_def_cfa reg7 +8
  .balign 4
LCieEnd:

LFdeHdr:
  .long LFdeEnd - LFdeStart
LFdeStart:
  .long LFdeStart - LCieHdr
  # The next two fields are longs instead of quads because of the sdata4
  # encoding.
  .long _main - .        # Function address
  .long LmainEnd - _main # Function length
  .byte 0
  ## Insert DW_CFA_GNU_args_size to prevent ld64 from creating a compact unwind
  ## entry to replace this FDE. Makes it easier for us to cross-check behavior
  ## across the two linkers (LLD never bothers trying to synthesize compact
  ## unwind if it is not already present).
  .byte 0x2e, 0x10       # DW_CFA_GNU_args_size
  .balign 4
LFdeEnd:

  .long 0 # terminator

.subsections_via_symbols 
