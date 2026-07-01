# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

## Check we can overlap a NOLOAD output section containing a NOBITS input section.
# RUN: ld.lld --script nobits.lds a.o -o out
# RUN: llvm-readelf -S -l out | FileCheck %s -check-prefix OVERLAP-NOBITS
# OVERLAP-NOBITS:      Name      Type     Address          Off               Size
# OVERLAP-NOBITS:      .bss      NOBITS   0000000000101000 [[OFF:[0-9a-f]+]] 001000
# OVERLAP-NOBITS-NEXT: .overlap  PROGBITS 0000000000200000 [[OFF]]           000001

# OVERLAP-NOBITS:      Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz
# OVERLAP-NOBITS:      LOAD 0x002000 0x0000000000101000 0x0000000000101000 0x000000 0x001000
# OVERLAP-NOBITS-NEXT: LOAD 0x002000 0x0000000000200000 0x0000000000101000 0x000001 0x000001

# OVERLAP-NOBITS: Section to Segment mapping:
# OVERLAP-NOBITS: 01 .bss
# OVERLAP-NOBITS: 02 .overlap

## Check we can overlap a NOLOAD output section containing a PROGBITS input section.
# RUN: ld.lld --script noload.lds a.o -o out
# RUN: llvm-readelf -S -l out | FileCheck %s -check-prefix OVERLAP-NOLOAD

# OVERLAP-NOLOAD:      Name      Type     Address          Off               Size
# OVERLAP-NOLOAD:      .data     NOBITS   0000000000101000 [[OFF:[0-9a-f]+]] 001000
# OVERLAP-NOLOAD-NEXT: .overlap  PROGBITS 0000000000200000 [[OFF]]           000001

# OVERLAP-NOLOAD:      Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz
# OVERLAP-NOLOAD:      LOAD 0x002000 0x0000000000101000 0x0000000000101000 0x000000 0x001000
# OVERLAP-NOLOAD-NEXT: LOAD 0x002000 0x0000000000200000 0x0000000000101000 0x000001 0x000001

# OVERLAP-NOLOAD:      Section to Segment mapping:
# OVERLAP-NOLOAD:      01 .data
# OVERLAP-NOLOAD-NEXT: 02 .overlap

## Check that we cannot overlap the memory occupied by a NOBITS input section
## that is assigned to a PROGBITS output section.
# RUN: not ld.lld --script progbits.lds a.o -o /dev/null 2>&1 | FileCheck %s -check-prefix ERR-PROGBITS-TYPE
# ERR-PROGBITS-TYPE: error: section .progbits load address range overlaps with .overlap

#--- a.s
.section .text,"ax",@progbits
  nop
.org 4096

.section .data,"aw",@progbits
.fill 4096, 1, 0xFF

.section .bss,"aw",@nobits
.zero 4096

.section .overlap,"aw",@progbits
.byte 1

#--- nobits.lds
MEMORY {
    RAM1 : ORIGIN = 0x100000, LENGTH = 1M
    RAM2 : ORIGIN = 0x200000, LENGTH = 1M
}

SECTIONS {
  .text : { *(.text); _start = .; } > RAM1
  .bss : { *(.bss); } > RAM1
  .overlap : AT(LOADADDR(.bss)) { *(.overlap); } > RAM2
  /DISCARD/ : { *(.data); *(.nobits); }
}

#--- noload.lds
MEMORY {
    RAM1 : ORIGIN = 0x100000, LENGTH = 1M
    RAM2 : ORIGIN = 0x200000, LENGTH = 1M
}

SECTIONS {
  .text : { *(.text); _start = .; } > RAM1
  .data (NOLOAD) : { *(.data); } > RAM1
  .overlap : AT(LOADADDR(.data)) { *(.overlap); } > RAM2
  /DISCARD/ : { *(.bss); *(.nobits); }
}

#--- progbits.lds
MEMORY {
    RAM1 : ORIGIN = 0x100000, LENGTH = 1M
    RAM2 : ORIGIN = 0x200000, LENGTH = 1M
}

SECTIONS {
  .progbits : { *(.text); _start = .; *(.bss); } > RAM1
  .overlap : AT(_start) { *(.overlap); } > RAM2
  /DISCARD/ : { *(.data); }
}
