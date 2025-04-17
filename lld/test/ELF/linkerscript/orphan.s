# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

## .jcr is a relro section and should be placed before other RW sections.
## .bss is SHT_NOBITS section and should be last RW section, so some space
## in ELF file could be saved.
# RUN: ld.lld a.o -T text-rw.lds -o text-rw
# RUN: llvm-readelf -S text-rw | FileCheck %s --check-prefix=TEXT-RW
# TEXT-RW:      .interp   PROGBITS 00000000000002{{..}} 0
# TEXT-RW-NEXT: .note.my  NOTE     00000000000002{{..}} 0
# TEXT-RW-NEXT: .text     PROGBITS 00000000000002{{..}} 0
# TEXT-RW-NEXT: .mytext   PROGBITS 00000000000002{{..}} 0
# TEXT-RW-NEXT: .jcr      PROGBITS 00000000000002{{..}} 0
# TEXT-RW-NEXT: .rw1      PROGBITS 0000000000001{{...}} 0
# TEXT-RW-NEXT: .rw2      PROGBITS 0000000000001{{...}} 0
# TEXT-RW-NEXT: .rw3      PROGBITS 0000000000001{{...}} 0
# TEXT-RW-NEXT: .bss      NOBITS   0000000000001{{...}} 0

# RUN: ld.lld a.o -T only-text.lds -o only-text
# RUN: llvm-readelf -S only-text | FileCheck %s --check-prefix=ONLY-TEXT
# ONLY-TEXT:      .interp   PROGBITS 00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .note.my  NOTE     00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .text     PROGBITS 00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .mytext   PROGBITS 00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .jcr      PROGBITS 00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .rw1      PROGBITS 00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .rw2      PROGBITS 00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .rw3      PROGBITS 00000000000002{{..}} 0
# ONLY-TEXT-NEXT: .bss      NOBITS   00000000000002{{..}} 0

# RUN: ld.lld a.o -T text-align.lds -o text-align
# RUN: llvm-readelf -S text-align | FileCheck %s --check-prefix=TEXT-ALIGN
# TEXT-ALIGN:      .interp   PROGBITS 00000000000002{{..}} 0
# TEXT-ALIGN-NEXT: .note.my  NOTE     00000000000002{{..}} 0
# TEXT-ALIGN-NEXT: .text     PROGBITS 00000000000002{{..}} 0
# TEXT-ALIGN-NEXT: .mytext   PROGBITS 0000000000001000     0
# TEXT-ALIGN-NEXT: .jcr      PROGBITS 0000000000001{{...}} 0
# TEXT-ALIGN-NEXT: .rw1      PROGBITS 0000000000001{{...}} 0
# TEXT-ALIGN-NEXT: .rw2      PROGBITS 0000000000001{{...}} 0
# TEXT-ALIGN-NEXT: .rw3      PROGBITS 0000000000001{{...}} 0
# TEXT-ALIGN-NEXT: .bss      NOBITS   0000000000001{{...}} 0

# RUN: ld.lld a.o -T only-rw.lds -o only-rw
# RUN: llvm-readelf -S only-rw | FileCheck %s --check-prefix=ONLY-RW
# ONLY-RW:         .interp   PROGBITS 00000000000002{{..}} 0
# ONLY-RW-NEXT:    .note.my  NOTE     00000000000002{{..}} 0
# ONLY-RW-NEXT:    .text     PROGBITS 00000000000002{{..}} 0
# ONLY-RW-NEXT:    .mytext   PROGBITS 00000000000002{{..}} 0
# ONLY-RW-NEXT:    .jcr      PROGBITS 00000000000002{{..}} 0
# ONLY-RW-NEXT:    .rw1      PROGBITS 00000000000002{{..}} 0
# ONLY-RW-NEXT:    .rw2      PROGBITS 0000000000001{{...}} 0
# ONLY-RW-NEXT:    .rw3      PROGBITS 0000000000001{{...}} 0
# ONLY-RW-NEXT:    .bss      NOBITS   0000000000001{{...}} 0

# RUN: ld.lld a.o -T rw-text.lds -o rw-text
# RUN: llvm-readelf -S rw-text | FileCheck %s --check-prefix=RW-TEXT
# RW-TEXT:      .jcr      PROGBITS 00000000000002{{..}} 0
# RW-TEXT-NEXT: .rw1      PROGBITS 00000000000002{{..}} 0
# RW-TEXT-NEXT: .rw2      PROGBITS 00000000000002{{..}} 0
# RW-TEXT-NEXT: .rw3      PROGBITS 00000000000002{{..}} 0
# RW-TEXT-NEXT: .bss      NOBITS   00000000000002{{..}} 0
# RW-TEXT-NEXT: .interp   PROGBITS 00000000000002{{..}} 0
# RW-TEXT-NEXT: .note.my  NOTE     00000000000002{{..}} 0
# RW-TEXT-NEXT: .text     PROGBITS 0000000000001{{...}} 0
# RW-TEXT-NEXT: .mytext   PROGBITS 0000000000001{{...}} 0

# RUN: ld.lld a.o -T rw-text-rw.lds -o rw-text-rw
# RUN: llvm-readelf -S rw-text-rw | FileCheck %s --check-prefix=RW-TEXT-RW
# RW-TEXT-RW:      .jcr      PROGBITS 00000000000002{{..}} 0
# RW-TEXT-RW-NEXT: .rw1      PROGBITS 00000000000002{{..}} 0
# RW-TEXT-RW-NEXT: .interp   PROGBITS 00000000000002{{..}} 0
# RW-TEXT-RW-NEXT: .note.my  NOTE     00000000000002{{..}} 0
# RW-TEXT-RW-NEXT: .text     PROGBITS 0000000000001{{...}} 0
# RW-TEXT-RW-NEXT: .mytext   PROGBITS 0000000000001{{...}} 0
# RW-TEXT-RW-NEXT: .rw2      PROGBITS 0000000000002{{...}} 0
# RW-TEXT-RW-NEXT: .rw3      PROGBITS 0000000000002{{...}} 0
# RW-TEXT-RW-NEXT: .bss      NOBITS   0000000000002{{...}} 0

#--- a.s
.section .rw1, "aw"; .byte 0
.section .rw2, "aw"; .byte 0
.section .rw3, "aw"; .byte 0
.section .jcr, "aw"; .byte 0
.section .bss, "aw",@nobits; .byte 0
.section .note.my, "a", @note; .byte 0
.section .interp, "a", @progbits; .byte 0
.text; nop
.section .mytext,"ax"; nop

#--- text-rw.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .text : { *(.text) }
  . = ALIGN(CONSTANT(MAXPAGESIZE));
  .rw1 : { *(.rw1) }
  .rw2 : { *(.rw2) }
}

#--- only-text.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .text : { *(.text) }
}

#--- text-align.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .text : { *(.text) }
  . = ALIGN(CONSTANT(MAXPAGESIZE));
}

#--- only-rw.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .rw1 : { *(.rw1) }
  . = ALIGN(CONSTANT(MAXPAGESIZE));
}

#--- rw-text.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .rw1 : { *(.rw1) }
  . = ALIGN(CONSTANT(MAXPAGESIZE));
  .text : { *(.text) }
}

#--- rw-text-rw.lds
SECTIONS {
  . = SIZEOF_HEADERS;
  .rw1 : { *(.rw1) }
  . = ALIGN(CONSTANT(MAXPAGESIZE));
  .text : { *(.text) }
  . = ALIGN(CONSTANT(MAXPAGESIZE));
  .rw2 : { *(.rw2) }
}
