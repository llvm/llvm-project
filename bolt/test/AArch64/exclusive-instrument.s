// This test checks that the foo function having exclusive memory access
// instructions won't be instrumented.

// REQUIRES: system-linux,bolt-runtime,target=aarch64{{.*}}

// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
// RUN:   %s -o %t.o
// RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib -Wl,-q -Wl,-fini=dummy
// RUN: llvm-bolt %t.exe -o %t.bolt -instrument -v=1 | FileCheck %s

// CHECK: Function foo has exclusive instructions, skip instrumentation

.global foo
.type foo, %function
foo:
  ldaxr w9, [x10]
  cbnz w9, .Lret
  stlxr w12, w11, [x9]
  cbz w12, foo
  clrex
.Lret:
  ret
.size foo, .-foo

.global _start
.type _start, %function
_start:
  cmp x0, #0
  b.eq .Lexit
  bl foo
.Lexit:
  ret
.size _start, .-_start

.global dummy
.type dummy, %function
dummy:
  ret
.size dummy, .-dummy
