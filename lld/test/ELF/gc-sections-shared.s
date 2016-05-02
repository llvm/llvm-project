# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --gc-sections --export-dynamic-symbol foo -o %t %t.o --as-needed %t2.so
# RUN: llvm-readobj --dynamic-table --dyn-symbols %t | FileCheck %s

# This test the property that we have a needed line for every undefined.
# It would also be OK to drop bar2 and the need for the .so


# CHECK: Name: bar
# CHECK: Name: bar2
# CHECK: Name: foo
# CHECK: NEEDED SharedLibrary ({{.*}}.so)


.section .text.foo, "ax"
.globl foo
foo:
call bar

.section .text.bar, "ax"
.globl bar
bar:
ret

.section .text._start, "ax"
.globl _start
_start:
ret

.section .text.unused, "ax"
call bar2
