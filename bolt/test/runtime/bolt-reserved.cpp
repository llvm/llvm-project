// REQUIRES: system-linux

/*
 * Check that llvm-bolt uses reserved space in a binary for allocating
 * new sections.
 */

// RUN: %clangxx %s -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt.exe 2>&1 | FileCheck %s
// RUN: %t.bolt.exe

// CHECK: BOLT-INFO: using reserved space

/*
 * Check that llvm-bolt detects a condition when the reserved space is
 * not enough for allocating new sections.
 */

// RUN: %clangxx %s -o %t.tiny.exe -Wl,--no-eh-frame-hdr -Wl,-q -DTINY
// RUN: not llvm-bolt %t.tiny.exe -o %t.tiny.bolt.exe 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-TINY

// CHECK-TINY: BOLT-ERROR: reserved space (1 byte) is smaller than required

#ifdef TINY
#define RSIZE "1"
#else
#define RSIZE "8192 * 1024"
#endif

asm(".pushsection .text \n\
       .globl __bolt_reserved_start \n\
       .type __bolt_reserved_start, @object \n\
       __bolt_reserved_start: \n\
       .space " RSIZE " \n\
       .globl __bolt_reserved_end \n\
       __bolt_reserved_end: \n\
     .popsection");

int main() { return 0; }
