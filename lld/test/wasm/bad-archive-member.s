# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux -o %t.dir/elf.o %s
# RUN: llvm-ar rcs %t.dir/libfoo.a %t.dir/elf.o
# RUN: not wasm-ld %t.dir/libfoo.a -o /dev/null 2>&1 | FileCheck %s
# CHECK: error: unknown file type: {{.*}}libfoo.a(elf.o)

.globl _start
_start:
