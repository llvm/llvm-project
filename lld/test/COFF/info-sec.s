// REQUIRES: x86
// Check that sections marked as IMAGE_SCN_LNK_INFO are excluded from the output.

// RUN: llvm-mc -filetype=obj -triple=x86_64-windows %s -o %t.obj
// RUN: lld-link -machine:amd64 -dll -noentry %t.obj -out:%t.dll
// RUN: llvm-readobj --headers %t.dll | FileCheck %s
// CHECK-NOT: Name: .test

        .section .test,"i"
        .long 1
