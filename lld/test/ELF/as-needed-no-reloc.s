# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -o %t %t.o --as-needed %t2.so
# RUN: llvm-readelf -d --dyn-symbols %t | FileCheck %s


# There must be a NEEDED entry for each undefined

# CHECK: (NEEDED) Shared library: [{{.*}}as-needed-no-reloc{{.*}}2.so]
# CHECK: UND bar

        .globl _start
_start:
        .global bar
