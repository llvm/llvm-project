// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/merge.s -o %t2.o
// RUN: ld.lld %t.o %t2.o -o %t
// RUN: llvm-readelf -S -s -x .mysec %t | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck --check-prefix=DISASM %s

// CHECK:      Name     Type     Address          Off    Size   ES Flg Lk Inf Al
// CHECK:      .mysec   PROGBITS 0000000000200120 000120 000008 04  AM  0   0  4

// CHECK-DAG:  0000000000200124     0 NOTYPE  LOCAL  DEFAULT     1 bar
// CHECK-DAG:  0000000000200124     0 NOTYPE  LOCAL  DEFAULT     1 zed
// CHECK-DAG:  0000000000200124     0 NOTYPE  LOCAL  HIDDEN      1 foo

// CHECK:      Hex dump of section '.mysec':
// CHECK-NEXT: 0x00200120 10000000 42000000

        .section        .mysec,"aM",@progbits,4
        .align  4
        .global foo
        .hidden foo
        .long   0x10
foo:
        .long   0x42
bar:
        .long   0x42
zed:
        .long   0x42

        .text
        .globl  _start
_start:
// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <_start>:

        movl .mysec, %eax
// addr(0x10) = 2097440
// DISASM-NEXT:   movl    2097440, %eax

        movl .mysec+7, %eax
// addr(0x42) + 3 = 2097444 + 3 = 2097447
// DISASM-NEXT:   movl    2097447, %eax

        movl .mysec+8, %eax
// addr(0x42) = 2097444
// DISASM-NEXT:   movl    2097444, %eax

        movl bar+7, %eax
// addr(0x42) + 7 = 2097444 + 7 = 2097451
// DISASM-NEXT:   movl    2097451, %eax

        movl bar+8, %eax
// addr(0x42) + 8 = 2097444 + 8 = 2097452
// DISASM-NEXT:   movl    2097452, %eax

        movl foo, %eax
// addr(0x42) = 2097444
// DISASM-NEXT:   movl    2097444, %eax

        movl foo+7, %eax
// addr(0x42) + 7 =  = 2097444 + 7 = 2097451
// DISASM-NEXT:   movl    2097451, %eax

        movl foo+8, %eax
// addr(0x42) + 8 =  = 2097444 + 8 = 2097452
// DISASM-NEXT:   movl    2097452, %eax

//  From the other file:  movl .mysec, %eax
// addr(0x42) = 2097444
// DISASM-NEXT:   movl    2097444, %eax
