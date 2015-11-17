// RUN: llvm-ar rc %t.a
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld2 -shared %t.o %t.a -o t
