// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s

// CHECK:  error: relocation R_X86_64_SIZE64 cannot be used against symbol 'foo'; recompile with -fPIC

        .global foo
foo:
        .quad 42
        .size foo, 8

        .quad foo@SIZE
