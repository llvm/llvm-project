// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/copy-in-shared.s -o %t2.o
// RUN: ld.lld -shared %t.o %t2.o -o %t.so

// Check that we ignore zero sized non-relro sections that are covered by the
// range of addresses covered by the PT_GNU_RELRO header.
// Check that we ignore zero sized relro sections that are disjoint from the
// range of addresses covered by the PT_GNU_RELRO header.
// REQUIRES: x86

// RUN: echo "SECTIONS { \
// RUN: .ctors : { *(.ctors) } \
// RUN: .large1 : { *(.large1) } \
// RUN: .dynamic : { *(.dynamic) } \
// RUN: .zero_size : { *(.zero_size) } \
// RUN: .jcr : { *(.jcr) } \
// RUN: .got.plt : { *(.got.plt) } \
// RUN: .large2 : { *(.large2) } \
// RUN: .data.rel.ro : { *(.data.rel.ro.*) } \
// RUN: } " > %t.script
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t3.o
// RUN: ld.lld %t3.o %t.so -o %t --script=%t.script
// RUN: llvm-readobj -program-headers %t | FileCheck %s

// CHECK: Type: PT_GNU_RELRO
// CHECK-NEXT: Offset:
// CHECK-NEXT: VirtualAddress:
// CHECK-NEXT: PhysicalAddress:
// CHECK-NEXT: FileSize:
// CHECK-NEXT: MemSize: 4096

        .section .text, "ax", @progbits
        .global _start
        .global bar
        .global foo
_start:
        callq bar

        // page size non-relro sections that would alter PT_GNU_RELRO header
        // MemSize if counted as part of relro.
        .section .large1, "aw", @progbits
        .space 4 * 1024

        .section .large2, "aw", @progbits
        .space 4 * 1024
        
        // empty relro section
        .section .ctors, "aw", @progbits
        
        // non-empty relro section
        .section .jcr, "aw", @progbits
        .quad 0

        // empty non-relro section
        .section .zero_size, "aw", @progbits
        .global sym
sym:

        // empty relro section
        .section .data.rel.ro, "aw", @progbits
        .global sym2
sym2:
