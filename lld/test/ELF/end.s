// Should set the value of the "_end" symbol to the end of the data segment.
// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

// By default, the .bss section is the latest section of the data segment.
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj -sections -symbols %t | FileCheck %s --check-prefix=DEFAULT

// DEFAULT: Sections [
// DEFAULT:     Name: .bss
// DEFAULT-NEXT:     Type:
// DEFAULT-NEXT:     Flags [
// DEFAULT-NEXT:       SHF_ALLOC
// DEFAULT-NEXT:       SHF_WRITE
// DEFAULT-NEXT:     ]
// DEFAULT-NEXT:     Address: 0x12002
// DEFAULT-NEXT:     Offset:
// DEFAULT-NEXT:     Size: 6
// DEFAULT: ]
// DEFAULT: Symbols [
// DEFAULT:     Name: _end
// DEFAULT-NEXT:     Value: 0x12008
// DEFAULT: ]

// If there is no .bss section, "_end" should point to the end of the .data section.
// RUN: echo "SECTIONS { \
// RUN:          /DISCARD/ : { *(.bss) } }" > %t.script
// RUN: ld.lld %t.o --script %t.script -o %t
// RUN: llvm-readobj -sections -symbols %t | FileCheck %s --check-prefix=NOBSS

// NOBSS: Sections [
// NOBSS:     Name: .data
// NOBSS-NEXT:     Type:
// NOBSS-NEXT:     Flags [
// NOBSS-NEXT:       SHF_ALLOC
// NOBSS-NEXT:       SHF_WRITE
// NOBSS-NEXT:     ]
// NOBSS-NEXT:     Address: 0x12000
// NOBSS-NEXT:     Offset:
// NOBSS-NEXT:     Size: 2
// NOBSS: ]
// NOBSS: Symbols [
// NOBSS:     Name: _end
// NOBSS-NEXT:     Value: 0x12002
// NOBSS: ]

// If the layout of the sections is changed, "_end" should point to the end of allocated address space.
// RUN: echo "SECTIONS { \
// RUN:          .bss : { *(.bss) } \
// RUN:          .data : { *(.data) } \
// RUN:          .text : { *(.text) } }" > %t.script
// RUN: ld.lld %t.o --script %t.script -o %t
// RUN: llvm-readobj -sections -symbols %t | FileCheck %s --check-prefix=TEXTATEND

// TEXTATEND: Sections [
// TEXTATEND:     Name: .text
// TEXTATEND-NEXT:     Type:
// TEXTATEND-NEXT:     Flags [
// TEXTATEND-NEXT:       SHF_ALLOC
// TEXTATEND-NEXT:       SHF_EXECINSTR
// TEXTATEND-NEXT:     ]
// TEXTATEND-NEXT:     Address: 0x12000
// TEXTATEND-NEXT:     Offset:
// TEXTATEND-NEXT:     Size: 1
// TEXTATEND: ]
// TEXTATEND: Symbols [
// TEXTATEND:     Name: _end
// TEXTATEND-NEXT:     Value: 0x12001
// TEXTATEND: ]

.global _start,_end
.text
_start:
    nop
.data
    .word 1
.bss
    .space 6
