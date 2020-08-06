# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# Empty SECTIONS command.
# RUN: echo "SECTIONS {}" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump --section-headers %t1 | \
# RUN:   FileCheck -check-prefix=SEC-DEFAULT %s

# SECTIONS command with the same order as default.
# RUN: echo "SECTIONS { \
# RUN:          .text : { *(.text) } \
# RUN:          .data : { *(.data) } }" > %t.script
# RUN: ld.lld -o %t2 --script %t.script %t
# RUN: llvm-objdump --section-headers %t2 | \
# RUN:   FileCheck -check-prefix=SEC-DEFAULT %s

#             Idx Name          Size
# SEC-DEFAULT: 1 .text         0000000e {{[0-9a-f]*}} TEXT
# SEC-DEFAULT: 2 .data         00000020 {{[0-9a-f]*}} DATA
# SEC-DEFAULT: 3 other         00000003 {{[0-9a-f]*}} DATA
# SEC-DEFAULT: 4 .bss          00000002 {{[0-9a-f]*}} BSS
# SEC-DEFAULT: 5 .comment      00000008 {{[0-9a-f]*}}
# SEC-DEFAULT: 6 .symtab       00000030 {{[0-9a-f]*}}
# SEC-DEFAULT: 7 .shstrtab     0000003b {{[0-9a-f]*}}
# SEC-DEFAULT: 8 .strtab       00000008 {{[0-9a-f]*}}

## Sections are placed in the order specified by the linker script. .data has
## a PT_LOAD segment, even if it is preceded by a non-alloc section. To
## allow this, place non-alloc orphan sections at the end and advance
## location counters for non-alloc non-orphan sections.
# RUN: echo "SECTIONS { \
# RUN:          .bss : { *(.bss) } \
# RUN:          other : { *(other) } \
# RUN:          .shstrtab : { *(.shstrtab) } \
# RUN:          .symtab : { *(.symtab) } \
# RUN:          .strtab : { *(.strtab) } \
# RUN:          .data : { *(.data) } \
# RUN:          .text : { *(.text) } }" > %t3.lds
# RUN: ld.lld -o %t3a -T %t3.lds %t
# RUN: llvm-readelf -S -l %t3a | FileCheck --check-prefix=SEC-ORDER %s
# RUN: ld.lld -o %t3b -T %t3.lds --unique %t
# RUN: llvm-readelf -S -l %t3b | FileCheck --check-prefix=SEC-ORDER %s

# SEC-ORDER:       [Nr] Name      Type     Address          Off    Size   ES Flg
# SEC-ORDER:       [ 0]           NULL     0000000000000000 000000 000000 00
# SEC-ORDER-NEXT:  [ 1] .bss      NOBITS   0000000000000000 001000 000002 00  WA
# SEC-ORDER-NEXT:  [ 2] other     PROGBITS 0000000000000002 001002 000003 00  WA
# SEC-ORDER-NEXT:  [ 3] .shstrtab STRTAB   0000000000000005 001005 00003b 00
# SEC-ORDER-NEXT:  [ 4] .symtab   SYMTAB   0000000000000040 001040 000030 18
# SEC-ORDER-NEXT:  [ 5] .strtab   STRTAB   0000000000000070 001070 000008 00
# SEC-ORDER-NEXT:  [ 6] .data     PROGBITS 0000000000000078 001078 000020 00  WA
# SEC-ORDER-NEXT:  [ 7] .text     PROGBITS 0000000000000098 001098 00000e 00  AX
# SEC-ORDER-NEXT:  [ 8] .comment  PROGBITS 0000000000000000 0010a6 000008 01  MS

# SEC-ORDER:        Type      Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# SEC-ORDER-NEXT:   LOAD      0x001000 0x0000000000000000 0x0000000000000000 0x000098 0x000098 RW  0x1000
# SEC-ORDER-NEXT:   LOAD      0x001098 0x0000000000000098 0x0000000000000098 0x00000e 0x00000e R E 0x1000
# SEC-ORDER-NEXT:   GNU_STACK 0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# .text and .data have swapped names but proper sizes and types.
# RUN: echo "SECTIONS { \
# RUN:          .data : { *(.text) } \
# RUN:          .text : { *(.data) } }" > %t.script
# RUN: ld.lld -o %t4 --script %t.script %t
# RUN: llvm-objdump --section-headers %t4 | \
# RUN:   FileCheck -check-prefix=SEC-SWAP-NAMES %s

#                Idx Name          Size
# SEC-SWAP-NAMES: 1 .data         0000000e {{[0-9a-f]*}} TEXT
# SEC-SWAP-NAMES: 2 .text         00000020 {{[0-9a-f]*}} DATA
# SEC-SWAP-NAMES: 3 other         00000003 {{[0-9a-f]*}} DATA
# SEC-SWAP-NAMES: 4 .bss          00000002 {{[0-9a-f]*}} BSS
# SEC-SWAP-NAMES: 5 .comment      00000008 {{[0-9a-f]*}}
# SEC-SWAP-NAMES: 6 .symtab       00000030 {{[0-9a-f]*}}
# SEC-SWAP-NAMES: 7 .shstrtab     0000003b {{[0-9a-f]*}}
# SEC-SWAP-NAMES: 8 .strtab       00000008 {{[0-9a-f]*}}

# Multiple SECTIONS command specifying additional input section descriptions
# for the same output section description - input sections are merged into
# one output section.
# RUN: echo "SECTIONS { \
# RUN:          .text : { *(.text) } \
# RUN:          .data : { *(.data) } } \
# RUN:       SECTIONS { \
# RUN:          .data : { *(other) } }" > %t.script
# RUN: ld.lld -o %t6 --script %t.script %t
# RUN: llvm-objdump --section-headers %t6 | \
# RUN:   FileCheck -check-prefix=SEC-MULTI %s

#           Idx Name          Size
# SEC-MULTI:      1 .text         0000000e {{[0-9a-f]*}} TEXT
# SEC-MULTI-NEXT:   .data         00000020 {{[0-9a-f]*}} DATA
# SEC-MULTI-NEXT:   .data         00000003 {{[0-9a-f]*}} DATA
# SEC-MULTI-NEXT:   .bss          00000002 {{[0-9a-f]*}} BSS
# SEC-MULTI-NEXT:   .comment      00000008 {{[0-9a-f]*}}
# SEC-MULTI-NEXT:   .symtab       00000030 {{[0-9a-f]*}}
# SEC-MULTI-NEXT:   .shstrtab     00000035 {{[0-9a-f]*}}
# SEC-MULTI-NEXT:   .strtab       00000008 {{[0-9a-f]*}}

## other is placed in a PT_LOAD segment even if it is preceded by a non-alloc section.
## The current implementation places .data, .bss, .comment and other in the same PT_LOAD.
# RUN: echo 'SECTIONS { \
# RUN:          .text : { *(.text) } \
# RUN:          .data : { *(.data) } \
# RUN:          .comment : { *(.comment) } \
# RUN:          other : { *(other) } }' > %t5.lds
# RUN: ld.lld -o %t5 -T %t5.lds %t
# RUN: llvm-readelf -S -l %t5 | FileCheck --check-prefix=SEP-BY-NONALLOC %s

# SEP-BY-NONALLOC:      [Nr] Name      Type     Address          Off    Size   ES Flg
# SEP-BY-NONALLOC:      [ 1] .text     PROGBITS 0000000000000000 001000 00000e 00  AX
# SEP-BY-NONALLOC-NEXT: [ 2] .data     PROGBITS 000000000000000e 00100e 000020 00  WA
# SEP-BY-NONALLOC-NEXT: [ 3] .bss      NOBITS   000000000000002e 00102e 000002 00  WA
# SEP-BY-NONALLOC-NEXT: [ 4] .comment  PROGBITS 0000000000000030 00102e 000008 01  MS
# SEP-BY-NONALLOC-NEXT: [ 5] other     PROGBITS 0000000000000038 001038 000003 00  WA

# SEP-BY-NONALLOC:      Type      Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# SEP-BY-NONALLOC-NEXT: LOAD      0x001000 0x0000000000000000 0x0000000000000000 0x00000e 0x00000e R E 0x1000
# SEP-BY-NONALLOC-NEXT: LOAD      0x00100e 0x000000000000000e 0x000000000000000e 0x00002d 0x00002d RW  0x1000
# SEP-BY-NONALLOC-NEXT: GNU_STACK 0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0

# Input section pattern contains additional semicolon.
# Case found in linux kernel script. Check we are able to parse it.
# RUN: echo "SECTIONS { .text : { ;;*(.text);;S = 0;; } }" > %t.script
# RUN: ld.lld -o /dev/null --script %t.script %t

.globl _start
_start:
    mov $60, %rax
    mov $42, %rdi

.section .data,"aw"
.quad 10, 10, 20, 20
.section other,"aw"
.short 10
.byte 20
.section .bss,"",@nobits
.short 0
