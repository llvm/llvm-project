// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/merge.s -o %t2.o
// RUN: ld.lld %t.o %t2.o -o %t
// RUN: llvm-readobj -s -section-data -t %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s

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

// CHECK:      Name: .mysec
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT:    Flags [
// CHECK-NEXT:      SHF_ALLOC
// CHECK-NEXT:      SHF_MERGE
// CHECK-NEXT:    ]
// CHECK-NEXT:    Address: 0x10120
// CHECK-NEXT:    Offset: 0x120
// CHECK-NEXT:    Size: 8
// CHECK-NEXT:    Link: 0
// CHECK-NEXT:    Info: 0
// CHECK-NEXT:    AddressAlignment: 4
// CHECK-NEXT:    EntrySize: 0
// CHECK-NEXT:    SectionData (
// CHECK-NEXT:      0000: 10000000 42000000
// CHECK-NEXT:    )


// Address of the constant 0x10 = 0x10120 = 65824
// Address of the constant 0x42 = 0x10124 = 65828

// CHECK:      Symbols [

// CHECK:        Name: bar
// CHECK-NEXT:   Value: 0x10124
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Loca
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .mysec

// CHECK:        Name: zed
// CHECK-NEXT:   Value: 0x10124
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .mysec

// CHECK:        Name: foo
// CHECK-NEXT:   Value: 0x10124
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 2
// CHECK-NEXT:   Section: .mysec

 // CHECK: ]

        .text
        .globl  _start
_start:
// DISASM:      Disassembly of section .text:
// DISASM-NEXT: _start:

        movl .mysec, %eax
// addr(0x10) = 65824
// DISASM-NEXT:   movl    65824, %eax

        movl .mysec+7, %eax
// addr(0x42) + 3 = 65828 + 3 = 65831
// DISASM-NEXT:   movl    65831, %eax

        movl .mysec+8, %eax
// addr(0x42) = 65828
// DISASM-NEXT:   movl    65828, %eax

        movl bar+7, %eax
// addr(0x42) + 7 = 65828 + 7 = 65835
// DISASM-NEXT:   movl    65835, %eax

        movl bar+8, %eax
// addr(0x42) + 8 = 65828 + 8 = 65836
// DISASM-NEXT:   movl    65836, %eax

        movl foo, %eax
// addr(0x42) = 65828
// DISASM-NEXT:   movl    65828, %eax

        movl foo+7, %eax
// addr(0x42) + 7 =  = 65828 + 7 = 65835
// DISASM-NEXT:   movl    65835, %eax

        movl foo+8, %eax
// addr(0x42) + 8 =  = 65828 + 8 = 65836
// DISASM-NEXT:   movl    65836, %eax

//  From the other file:  movl .mysec, %eax
// addr(0x42) = 65828
// DISASM-NEXT:   movl    65828, %eax
