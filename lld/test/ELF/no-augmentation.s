// Input file generated on a mips64 device running FreeBSD 11-CURRENT
// using the system compiler GCC 4.2.1, from no-augmentation.c containing:
// int fn(int a) { return a + 1; }
// with command:
// cc -funwind-tables -g -O0 -c no-augmentation.c

// RUN: llvm-mc -filetype=obj -triple=mips64-unknown-freebsd %s -o %t.o
// RUN: ld.lld --eh-frame-hdr %t.o %p/Inputs/no-augmentation.o -o %t \
// RUN:   | FileCheck -allow-empty %s

// REQUIRES: mips

// CHECK-NOT: corrupted or unsupported CIE information
// CHECK-NOT: corrupted CIE

.global __start
__start:
