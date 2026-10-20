// Verify that __guard_flags is set to CF_INSTRUMENTED if CF guard is disabled,
// but the input object was built with CF guard.

// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-windows %s -o %t.obj
// RUN: lld-link -out:%t1.dll %t.obj -dll -noentry
// RUN: lld-link -out:%t2.dll %t.obj -dll -noentry -guard:no

// RUN: llvm-readobj --hex-dump=.test %t1.dll | FileCheck %s
// RUN: llvm-readobj --hex-dump=.test %t2.dll | FileCheck %s
// CHECK: 0x180001000 00010000

        .def     @feat.00;
        .scl    3;
        .type   0;
        .endef
        .globl  @feat.00
@feat.00 = 0x800

        .section .test, "r"
        .long __guard_flags
