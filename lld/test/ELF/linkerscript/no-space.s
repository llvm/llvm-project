# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

# RUN: echo "SECTIONS {foo 0 : {*(foo*)} }" > %t.script
# RUN: not ld.lld -o %t --script %t.script %t.o -shared 2>&1 | FileCheck %s

# CHECK: Not enough space for ELF and program headers

.section foo, "a"
.quad 0
