; REQUIRES: x86_64
; RUN: echo 'int main() { return 0; }' > %t.c
; RUN: %clang -c -o %t.o %t.c
; RUN: %ld.lld -shared %t.o /usr/lib/x86_64-linux-gnu/Scrt1.o --dynamic-linker=/lib64/ld-linux-x86-64.so.2 -o %t.so
; RUN: llvm-readobj --program-headers %t.so | FileCheck %s
; CHECK: PT_INTERP
