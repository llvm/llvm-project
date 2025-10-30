// REQUIRES: x86-registered-target
// RUN: %clang -cc1as -triple x86_64 %s -filetype obj --gsframe -o %t.o
// RUN: llvm-readelf -S %t.o | FileCheck %s

// CHECK: .sframe
.cfi_startproc        
        call foo
.cfi_endproc
