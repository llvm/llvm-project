# Test that BOLT errs when trying to instrument a binary with a different
# architecture than the one BOLT is built for.

# REQUIRES: x86_64-linux,bolt-runtime,target=x86_64{{.*}}

# RUN: llvm-mc -triple aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld -q -pie -o %t.exe %t.o
# RUN: not llvm-bolt --instrument -o %t.out %t.exe 2>&1 | FileCheck %s

# CHECK: BOLT-ERROR: linking object with arch x86_64 into context with arch aarch64

    .text
    .globl _start
    .type _start, %function
_start:
    # BOLT errs when instrumenting without relocations; create a dummy one.
    .reloc 0, R_AARCH64_NONE
    ret
    .size _start, .-_start

    .globl _fini
    .type _fini, %function
    # Force DT_FINI to be created (needed for instrumentation).
_fini:
    ret
    .size _fini, .-_fini
