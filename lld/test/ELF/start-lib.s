// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
// RUN:   %p/Inputs/whole-archive.s -o %t2.o

// RUN: ld.lld -o %t3 %t1.o %t2.o
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=ADDED %s
// ADDED: Name: _bar

// RUN: ld.lld -o %t3 %t1.o --start-lib %t2.o
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=LIB %s
// LIB-NOT: Name: _bar

.globl _start
_start:
