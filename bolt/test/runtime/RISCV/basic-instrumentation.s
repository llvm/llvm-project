# REQUIRES: system-linux,bolt-runtime

# RUN: %clang %cflags -Wl,-q -o %t.exe %s
# RUN: llvm-bolt --instrument --instrumentation-file=%t.fdata -o %t.instr %t.exe

## Run the profiled binary and check that the profile reports at least that `f`
## has been called.
# RUN: rm -f %t.fdata
# RUN: %t.instr
# RUN: cat %t.fdata | FileCheck %s
# CHECK: f 0 0 1{{$}}

## Check BOLT works with this profile
# RUN: llvm-bolt --data %t.fdata --reorder-blocks=cache -o %t.bolt %t.exe

    .text
    .globl main
    .type main, @function
main:
    addi sp, sp, -8
    sd ra, 0(sp)
    call f
    ld ra, 0(sp)
    addi sp, sp, 8
    li a0, 0
    ret
    .size main, .-main

    .globl f
    .type f, @function
f:
    ret
    .size f, .-f
