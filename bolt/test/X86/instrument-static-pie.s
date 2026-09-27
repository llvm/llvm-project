# Test that BOLT will instrument a statically linked PIE using entry point
# patching.

# REQUIRES: system-linux,bolt-runtime,target=x86_64-{{.*}}

# RUN: llvm-mc -triple x86_64 -filetype=obj %s -o %t.o
# Using lld directly and not specifying and interpreter effectlively generates a static pie.
# RUN: ld.lld -q -pie -o %t.exe %t.o
# RUN: llvm-bolt --instrument --instrumentation-sleep-time=1 \
# RUN:   -o %t.out %t.exe 2>&1 | FileCheck %s

# CHECK: static pie executable detected
# CHECK: runtime library initialization was hooked via ELF Header Entry Point

    .text
    .globl _start
    .type _start, %function
_start:
    # BOLT errs when instrumenting without relocations; create a dummy one.
    .reloc 0, R_X86_64_NONE
    retq
    .size _start, .-_start

    .globl _init
    .type _init, %function
_init:
    ret
    .size _init, .-_init

    .globl _fini
    .type _fini, %function
_fini:
    ret
    .size _fini, .-_fini
