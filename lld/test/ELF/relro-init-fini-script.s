// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-none-elf %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:  .dynamic 0x10000 : { *(.dynamic) } \
// RUN:  .preinit_array : { PROVIDE_HIDDEN (__preinit_array_start = .); \
// RUN:                  KEEP (*(.preinit_array)) } \
// RUN:  .init_array : { PROVIDE_HIDDEN (__init_array_start = .); \
// RUN:                  KEEP (*(.init_array)) } \
// RUN:  .fini_array : { PROVIDE_HIDDEN (__fini_array_start = .); \
// RUN:                  KEEP (*(.fini_array)) } \
// RUN:  .data.rel.ro : { *(.data.rel.ro) } \
// RUN:  .data : { *(.data) } } " > %t.script
// RUN: ld.lld %t.o -o %t.so --shared --script=%t.script
// RUN: llvm-readelf -S %t.so | FileCheck %s
// RUN: llvm-readobj --segments %t.so | FileCheck %s --check-prefix=PHDR

/// Check that an empty .init_array, .fini_array or .preinit_array that is
/// kept due to symbol references, is still counted as RELRO. The _array
/// sections are zero size. The RELRO extent is [.dynamic, .data.rel.ro)

// CHECK:      .dynamic       DYNAMIC         0000000000010000 002000 000110
// CHECK-NEXT: .preinit_array PROGBITS        {{0+}}[[# %x,ADDR:]]
// CHECK-NEXT: .init_array    PROGBITS        {{0+}}[[# ADDR]]
// CHECK-NEXT: .fini_array    PROGBITS        {{0+}}[[# ADDR]]
// CHECK-NEXT: .data.rel.ro   PROGBITS        0000000000010110 002110 000008

// PHDR:      Type: PT_GNU_RELRO
// PHDR-NEXT: Offset: 0x2000
// PHDR-NEXT: VirtualAddress: 0x10000
// PHDR-NEXT: PhysicalAddress: 0x10000
// PHDR-NEXT: FileSize: 280
 .section .data.rel.ro, "aw", %progbits
 .global foo
 .quad foo

 .data
 .quad __init_array_start
 .quad __fini_array_start
 .quad __preinit_array_start
