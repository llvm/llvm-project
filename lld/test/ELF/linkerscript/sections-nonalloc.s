# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t.o

## Non-SHF_ALLOC sections are placed after all SHF_ALLOC sections. They will
## thus not be contained in a PT_LOAD segment. data2 has a PT_LOAD segment,
## even if it is preceded by a non-SHF_ALLOC section. Non-SHF_ALLOC orphan
## sections have zero addresses.
## NOTE: GNU ld assigns non-zero addresses to non-SHF_ALLOC non-orphan sections.
# RUN: ld.lld -T %t/a.lds %t.o -o %ta
# RUN: llvm-readelf -S -l %ta | FileCheck %s

# CHECK:       [Nr] Name      Type     Address          Off    Size   ES Flg Lk
# CHECK-NEXT:  [ 0]           NULL     0000000000000000 000000 000000 00      0
# CHECK-NEXT:  [ 1] .bss      NOBITS   0000000000000000 001000 000001 00  WA  0
# CHECK-NEXT:  [ 2] data1     PROGBITS 0000000000000001 001001 000001 00  WA  0
# CHECK-NEXT:  [ 3] data3     PROGBITS 0000000000000002 001002 000001 00  WA  0
# CHECK-NEXT:  [ 4] other1    PROGBITS 0000000000000000 001008 000001 00      0
# CHECK-NEXT:  [ 5] other2    PROGBITS 0000000000000000 001010 000001 00      0
## Orphan placement places other3, .symtab, .shstrtab and .strtab after other2.
# CHECK-NEXT:  [ 6] other3    PROGBITS 0000000000000000 001020 000001 00      0
# CHECK-NEXT:  [ 7] .symtab   SYMTAB   0000000000000000 001028 000030 18      9
# CHECK-NEXT:  [ 8] .shstrtab STRTAB   0000000000000000 001058 00004d 00      0
# CHECK-NEXT:  [ 9] .strtab   STRTAB   0000000000000000 0010a5 000008 00      0
# CHECK-NEXT:  [10] data2     PROGBITS 0000000000000003 001003 000001 00  WA  0
# CHECK-NEXT:  [11] .text     PROGBITS 0000000000000004 001004 000001 00  AX  0

# CHECK:       Type       Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT:  LOAD       0x001000 0x0000000000000000 0x0000000000000000 0x000004 0x000004 RW  0x1000
# CHECK-NEXT:  LOAD       0x001004 0x0000000000000004 0x0000000000000004 0x000001 0x000001 R E 0x1000
# CHECK-NEXT:  GNU_STACK  0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# RUN: ld.lld -T %t/b.lds %t.o -o %tb
# RUN: llvm-readelf -S -l %tb | FileCheck %s --check-prefix=CHECK1

# CHECK1:      [Nr] Name      Type     Address          Off    Size   ES Flg Lk
# CHECK1-NEXT: [ 0]           NULL     0000000000000000 000000 000000 00      0
# CHECK1-NEXT: [ 1] .text     PROGBITS 00000000000000b0 0000b0 000001 00  AX  0
# CHECK1-NEXT: [ 2] .bss      NOBITS   00000000000000b1 0000b1 000001 00  WA  0
# CHECK1-NEXT: [ 3] data1     PROGBITS 00000000000000b2 0000b2 000001 00  WA  0
# CHECK1-NEXT: [ 4] data3     PROGBITS 00000000000000b3 0000b3 000001 00  WA  0
# CHECK1-NEXT: [ 5] other1    PROGBITS 0000000000000000 0000b8 000001 00      0
# CHECK1-NEXT: [ 6] other2    PROGBITS 0000000000000000 0000c0 000001 00      0
# CHECK1-NEXT: [ 7] other3    PROGBITS 0000000000000000 0000d0 000001 00      0
# CHECK1-NEXT: [ 8] .symtab   SYMTAB   0000000000000000 0000d8 000030 18     10
# CHECK1-NEXT: [ 9] .shstrtab STRTAB   0000000000000000 000108 00004d 00      0
# CHECK1-NEXT: [10] .strtab   STRTAB   0000000000000000 000155 000008 00      0
# CHECK1-NEXT: [11] data2     PROGBITS 00000000000000b4 0000b4 000001 00  WA  0
# CHECK1:      Type       Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK1-NEXT: LOAD       0x000000 0x0000000000000000 0x0000000000000000 0x0000b5 0x0000b5 RWE 0x1000
# CHECK1-NEXT: 0x60000000 0x0000b8 0x0000000000000000 0x0000000000000000 0x000009 0x000001     0x8

#--- a.lds
SECTIONS {
  .bss : { *(.bss) }
  data1 : { *(data1) }
  other1 : { *(other1) }
  other2 : { *(other2) }
  data2 : { *(data2) }
  .text : { *(.text) }
  /DISCARD/ : { *(.comment) }
}

#--- b.lds
PHDRS {
  text PT_LOAD FILEHDR PHDRS;
  foo 0x60000000 FLAGS (0);
}
SECTIONS {
  . = SIZEOF_HEADERS;
  .text : { *(.text) } : text
  .bss : { *(.bss) } : text
  data1 : { *(data1) } : text
  other1 : { *(other1) } : foo
  other2 : { *(other2) } : foo
  data2 : { *(data1) } : text
  /DISCARD/ : { *(.comment) }
}

#--- main.s
.globl _start
_start: nop
.section data1,"aw"; .byte 0
.section data2,"aw"; .byte 0
.section data3,"aw"; .byte 0
.bss; .byte 0

.section other1; .p2align 2; .byte 0
.section other2; .p2align 3; .byte 0
.section other3; .p2align 4; .byte 0
