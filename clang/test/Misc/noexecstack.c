// REQUIRES: x86-registered-target
// RUN: %clang -cc1 -triple x86_64 %s -emit-obj -o %t.o -mnoexecstack
// RUN: llvm-readelf -S %t.o | FileCheck %s

// RUN: %clang -cc1 -triple x86_64 %s -S -o %t.s
// RUN: FileCheck --check-prefix=ASM %s < %t.s
// RUN: %clang -cc1as -triple x86_64 %t.s -filetype obj -mnoexecstack -o %t.o
// RUN: llvm-readelf -S %t.o | FileCheck %s

// CHECK: .text             PROGBITS        0000000000000000 {{[0-9a-f]+}} 000001 00  AX  0   0 16
// CHECK: .note.GNU-stack   PROGBITS        0000000000000000 {{[0-9a-f]+}} 000000 00      0   0  1

// ASM:     .text
// ASM:     .section        ".note.GNU-stack","",@progbits
// ASM-NOT: ".note.GNU-stack"

void f() {}
