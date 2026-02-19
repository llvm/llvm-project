// REQUIRES: x86-registered-target
// RUN: %clang -cc1as -triple x86_64 %s -filetype obj --gsframe | llvm-readelf -S - | FileCheck %s

// CHECK: .sframe
.cfi_startproc
        call foo
.cfi_endproc
