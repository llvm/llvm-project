# REQUIRES: x86
## If neither AT(lma) nor AT>lma_region is specified, don't propagate
## lmaOffset if the section and the previous section are in different memory
## regions.

# RUN: echo '.globl _start; _start: ret; \
# RUN:   .data; .byte 0; \
# RUN:   .bss; .byte 0' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t.o
# RUN: ld.lld -T %s %t.o -o %t
# RUN: llvm-readelf -l %t | FileCheck %s

## GNU ld places .text and .bss in the same RWX PT_LOAD.
# CHECK:      Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg
# CHECK-NEXT: LOAD 0x001000 0x000000000fb00000 0x000000000fb00000 0x000001 0x000001 R E
# CHECK-NEXT: LOAD 0x002000 0x000000001f400000 0x000000000fb01000 0x000001 0x000001 RW
# CHECK-NEXT: LOAD 0x002001 0x000000000fb00001 0x000000000fb00001 0x000000 0x000001 RW

MEMORY  {
  DDR : o = 0xfb00000, l = 185M
  TCM : o = 0x1f400000, l = 128K
}

SECTIONS {
  .text : { *(.text) } > DDR
  .mdata : AT(0xfb01000) { *(.data); } > TCM
  ## .mdata and .bss are in different memory regions. Start a new PT_LOAD for
  ## .bss, even if .mdata does not set a LMA region.
  .bss : { *(.bss) } > DDR
}
