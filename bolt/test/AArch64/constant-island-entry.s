// This test checks that we ignore functions which add an entry point that
// is in a constant island.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -pie -Wl,-q -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: Ignoring entry point at address 0x{{[0-9a-f]+}} in constant island of function func

.globl func
.type func, %function
func:
  b .Lafter_constant

.type constant_island, %object
constant_island:
  .xword 0xabcdef

.Lafter_constant:
  ret
  .size func, .-func

.globl caller
.type caller, %function
caller:
  bl constant_island
  ret
