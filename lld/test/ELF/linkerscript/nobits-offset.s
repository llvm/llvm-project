# REQUIRES: x86
## Test file offsets of SHT_NOBITS sections and of the sections following them
## in the same PT_LOAD.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o

## If a SHT_NOBITS section is the only section of a PT_LOAD segment,
## p_offset will be set to the sh_offset field of the section. Check we align
## sh_offset to sh_addr modulo max-page-size, so that p_vaddr=p_offset (mod
## p_align).
# RUN: ld.lld a.o -T a.lds -o a
# RUN: llvm-readelf -S -l a | FileCheck %s

# CHECK:      Name  Type     Address          Off     Size
# CHECK-NEXT:       NULL     0000000000000000 000000  000000
# CHECK-NEXT: .text PROGBITS 0000000000000000 000190  000000
# CHECK-NEXT: .sec1 NOBITS   0000000000000000 001000  000001
# CHECK-NEXT: .bss  NOBITS   0000000000000400 001400  000001

# CHECK:      Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT: LOAD 0x001000 0x0000000000000000 0x0000000000000000 0x000000 0x000001 R   0x1000
# CHECK-NEXT: LOAD 0x001400 0x0000000000000400 0x0000000000000400 0x000000 0x000001 RW  0x1000

# CHECK:      00 .sec1 {{$}}
# CHECK:      01 .bss {{$}}

# RUN: ld.lld b.o -T b.lds -o b
# RUN: llvm-readelf -S -l b | FileCheck %s --check-prefix=EMPTY
# RUN: ld.lld b.o -T c.lds -o c
# RUN: llvm-readelf -S -l c | FileCheck %s --check-prefix=NONEMPTY

# EMPTY:         .bss    NOBITS   0000000000000002 001002 000100
# EMPTY-NEXT:    .empty1 PROGBITS 0000000000000102 001102 000000
# EMPTY-NEXT:    .empty2 PROGBITS 0000000000000102 001102 000000
# EMPTY:         LOAD 0x001001 0x0000000000000001 0x0000000000000001 0x000101 0x000101 RW 0x1000

# NONEMPTY:      .bss    NOBITS   0000000000000002 001002 000100
# NONEMPTY-NEXT: .empty1 PROGBITS 0000000000000102 001102 000000
# NONEMPTY-NEXT: .data2  PROGBITS 0000000000000102 001102 000001
# NONEMPTY:      LOAD 0x001001 0x0000000000000001 0x0000000000000001 0x000102 0x000102 RW 0x1000

#--- a.s
.bss
.p2align 10
.byte 0

#--- a.lds
SECTIONS {
  .sec1 (NOLOAD) : { . += 1; }
  .bss : { *(.bss) }
}

#--- b.s
.globl _start
_start: ret
.data
.byte 1
.bss
.space 0x100

#--- b.lds
SECTIONS {
  .data : {}
  .bss : {}
  .empty1 : { empty1 = .; }
  .empty2 : { empty2 = .; }
}

#--- c.lds
SECTIONS {
  .data : {}
  .bss : {}
  .empty1 : { empty1 = .; }
  .data2 : { BYTE(2) }
}
