# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-freebsd %s -o %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-freebsd \
# RUN:   %p/Inputs/libsearch-dyn.s -o %tdyn.o
# RUN: mkdir -p %t.dir
# RUN: ld.lld -shared %tdyn.o -o /dev/null
# RUN: echo "SEARCH_DIR(\"%t.dir\")" > %t.script
# RUN: ld.lld -o /dev/null --script %t.script -lls %t

.globl _start,_bar
_start:
