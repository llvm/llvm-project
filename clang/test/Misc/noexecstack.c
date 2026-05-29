// REQUIRES: x86-registered-target
// RUN: %clang -cc1 -triple x86_64 %s -emit-obj -o %t.o -mnoexecstack
// RUN: llvm-readelf -S %t.o | FileCheck %s

// RUN: %clang -cc1 -triple x86_64 %s -S -o %t.s
// RUN: FileCheck --check-prefix=ASM %s < %t.s
// RUN: %clang -cc1as -triple x86_64 %t.s -filetype obj -mnoexecstack -o %t.o
// RUN: llvm-readelf -S %t.o | FileCheck %s
/// Without -mnoexecstack on a .s that lacks .note.GNU-stack, the section should be absent.
// RUN: echo "nop" | %clang -cc1as -triple x86_64 - -filetype obj -o %t.o
// RUN: llvm-readelf -S %t.o | FileCheck --check-prefix=NOSTACK %s

// CHECK: .text             PROGBITS        0000000000000000 {{[0-9a-f]+}} 000001 00  AX  0   0  4
// CHECK: .note.GNU-stack   PROGBITS        0000000000000000 {{[0-9a-f]+}} 000000 00      0   0  1

// NOSTACK-NOT: .note.GNU-stack

// ASM:     .text
// ASM:     .section        ".note.GNU-stack","",@progbits
// ASM-NOT: ".note.GNU-stack"

void f() {}
