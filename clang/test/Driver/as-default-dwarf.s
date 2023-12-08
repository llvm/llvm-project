@ REQUIRES: arm-registered-target
@ RUN: %clang --target=armv8a--linux-gnueabi -c %s -o %t
@ RUN: llvm-objdump -t %t | FileCheck %s
    .text
    .type   foo,%function
foo:
    .fnstart
    .cfi_startproc

.Ltmp2:
    .size   foo, .Ltmp2-foo
    .cfi_endproc
    .fnend
    .cfi_sections .debug_frame
@ CHECK: foo
