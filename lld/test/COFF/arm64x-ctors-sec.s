// REQUIRES: aarch64, x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows ctor1-arm64.s -o ctor1-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows ctor2-arm64.s -o ctor2-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows ctor1-arm64ec.s -o ctor1-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows ctor2-amd64.s -o ctor2-amd64.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows test.s -o test-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test.s -o test-arm64ec.obj

// Check that .ctors and .dtors chunks are correctly sorted and that EC and native chunks are split.

// RUN: lld-link -out:out.dll -machine:arm64x -lldmingw -dll -noentry test-arm64.obj test-arm64ec.obj \
// RUN:           ctor1-arm64.obj ctor2-arm64.obj ctor1-arm64ec.obj ctor2-amd64.obj
// RUN: llvm-readobj --hex-dump=.rdata --hex-dump=.test out.dll | FileCheck %s

// RUN: lld-link -out:out2.dll -machine:arm64x -lldmingw -dll -noentry test-arm64.obj test-arm64ec.obj \
// RUN:           ctor1-arm64ec.obj ctor2-amd64.obj ctor1-arm64.obj ctor2-arm64.obj
// RUN: llvm-readobj --hex-dump=.rdata --hex-dump=.test out2.dll | FileCheck %s

// RUN: lld-link -out:out3.dll -machine:arm64x -lldmingw -dll -noentry test-arm64.obj test-arm64ec.obj \
// RUN:           ctor2-arm64.obj ctor1-arm64ec.obj ctor2-amd64.obj ctor1-arm64.obj
// RUN: llvm-readobj --hex-dump=.rdata --hex-dump=.test out3.dll | FileCheck %s

// CHECK:      Hex dump of section '.rdata':
// CHECK-NEXT: 0x180001000 ffffffff ffffffff 01000000 00000000
// CHECK-NEXT: 0x180001010 02000000 00000000 03000000 00000000
// CHECK-NEXT: 0x180001020 00000000 00000000 ffffffff ffffffff
// CHECK-NEXT: 0x180001030 11000000 00000000 12000000 00000000
// CHECK-NEXT: 0x180001040 13000000 00000000 00000000 00000000
// CHECK-NEXT: 0x180001050 ffffffff ffffffff 01010000 00000000
// CHECK-NEXT: 0x180001060 02010000 00000000 03010000 00000000
// CHECK-NEXT: 0x180001070 00000000 00000000 ffffffff ffffffff
// CHECK-NEXT: 0x180001080 11010000 00000000 12010000 00000000
// CHECK-NEXT: 0x180001090 13010000 00000000 00000000 00000000
// CHECK-EMPTY:
// CHECK-NEXT: Hex dump of section '.test':
// CHECK-NEXT: 0x180003000 00100000 50100000 28100000 78100000

#--- ctor1-arm64.s
        .section .ctors.1,"drw"
        .xword 1
        .section .ctors.3,"drw"
        .xword 3
        .section .dtors.1,"drw"
        .xword 0x101
        .section .dtors.3,"drw"
        .xword 0x103

#--- ctor2-arm64.s
        .section .ctors.2,"drw"
        .xword 2
        .section .dtors.2,"drw"
        .xword 0x102

#--- ctor1-arm64ec.s
        .section .ctors.1,"drw"
        .xword 0x11
        .section .ctors.3,"drw"
        .xword 0x13
        .section .dtors.1,"drw"
        .xword 0x111
        .section .dtors.3,"drw"
        .xword 0x113

#--- ctor2-amd64.s
        .section .ctors.2,"drw"
        .quad 0x12
        .section .dtors.2,"drw"
        .quad 0x112

#--- test.s
        .section .test
        .rva __CTOR_LIST__
        .rva __DTOR_LIST__

