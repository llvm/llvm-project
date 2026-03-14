# REQUIRES: x86
# Tests error on archive file without a symbol table
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux -o %t.o %s
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux -o %t.archive.o %S/Inputs/archive.s
# RUN: rm -f %t.a
# RUN: llvm-ar crS %t.a %t.archive.o

# RUN: ld.lld %t.o %t.a -o /dev/null 2>&1 | count 0

# RUN: ld.lld -shared %t.archive.o -o %t.so
# RUN: llvm-ar crS %t.a %t.so
# RUN: ld.lld %t.o %t.a -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN

# WARN: warning: {{.*}}.a: archive member '{{.*}}.so' is neither ET_REL nor LLVM bitcode

.globl _start
_start:
