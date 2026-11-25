// This test checks that the foo function having exclusive memory access
// instructions won't be instrumented.

// REQUIRES: system-linux,bolt-runtime,target=aarch64{{.*}}

// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
// RUN:   %s -o %t.o
// RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib -Wl,-q -Wl,-fini=dummy
// RUN: llvm-bolt %t.exe -o %t.bolt -instrument -v=2 | FileCheck %s

// Counters are kept out of an exclusive region even when the block holding
// the store is a spanning tree leaf, which gets a block counter instead of
// edge counters.
// RUN: llvm-bolt %t.exe -o %t.leaf.bolt -instrument --print-finalized \
// RUN:   --print-only=case6 | FileCheck %s --check-prefix=CHECK-LEAF
// CHECK-LEAF:     ldaxr
// CHECK-LEAF-NOT: stadd
// CHECK-LEAF:     stlxr
// CHECK-LEAF:     stadd

// A spanning tree leaf that starts an exclusive region keeps its counter,
// which runs before the exclusive load.
// RUN: llvm-bolt %t.exe -o %t.entry.bolt -instrument --print-finalized \
// RUN:   --print-only=case7 | FileCheck %s --check-prefix=CHECK-ENTRY
// CHECK-ENTRY:      stadd
// CHECK-ENTRY-NEXT: ldp
// CHECK-ENTRY-NEXT: ldaxr
// CHECK-ENTRY-NOT:  stadd
// CHECK-ENTRY:      stlxr

// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function foo
// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function foo
// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function foo
// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function case1
// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function case2
// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function case2
// CHECK: BOLT-INSTRUMENTER: function case3 has exclusive store without corresponding load. Ignoring the function.
// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function case4
// CHECK: BOLT-INSTRUMENTER: function case4 has two exclusive loads. Ignoring the function.
// CHECK: BOLT-INSTRUMENTER: skip BB {{.*}} due to exclusive instruction in function case5
// CHECK: BOLT-INSTRUMENTER: function case5 has exclusive load in trailing BB. Ignoring the function.

.global foo
.type foo, %function
foo:
  # exclusive load and store in two bbs
  ldaxr w9, [x10]
  cbnz w9, .Lret
  stlxr w12, w11, [x9]
  cbz w12, foo
.Lret:
  clrex
  ret
.size foo, .-foo

.global _start
.type _start, %function
_start:
  mov x0, #0
  mov x1, #1
  mov x2, #2
  mov x3, #3

  bl case1
  bl case2
  bl case3
  bl case4
  bl case5
  bl case6
  bl case7

  ret
.size _start, .-_start

# Case 1: exclusive load and store in one basic block
.global case1
.type case1, %function
case1:
  str  x0, [x2]
  ldxr w0, [x2]
  add  w0, w0, #1
  stxr w1, w0, [x2]
  ret
.size case1, .-case1

# Case 2: exclusive load and store in different blocks
.global case2
.type case2, %function
case2:
  b    case2_load

case2_load:
  ldxr x0, [x2]
  b    case2_store

case2_store:
  add  x0, x0, #1
  stxr w1, x0, [x2]
  ret
.size case2, .-case2

# Case 3: store without preceding load
.global case3
.type case3, %function
case3:
  stxr w1, x3, [x2]
  ret
.size case3, .-case3

# Case 4: two exclusive load instructions in neighboring blocks
.global case4
.type case4, %function
case4:
  b    case4_load

case4_load:
  ldxr x0, [x2]
  b    case4_load_next

case4_load_next:
  ldxr x1, [x2]
  ret
.size case4, .-case4

# Case 5:  Exclusive load without successor
.global case5
.type case5, %function
case5:
  ldxr x0, [x2]
  ret
.size case5, .-case5

# Case 6: the block holding the exclusive store is a spanning tree leaf
.global case6
.type case6, %function
case6:
  ldaxr w0, [x1]
  cbz  w0, .Lcase6_store
  clrex
  b    .Lcase6_done
.Lcase6_store:
  stlxr w2, w3, [x1]
  cbnz w2, case6
.Lcase6_done:
  ret
.size case6, .-case6

# Case 7: a block that starts an exclusive region is a spanning tree leaf
.global case7
.type case7, %function
case7:
  cbz  x0, .Lcase7_load
  ldaxr w0, [x1]
  b    .Lcase7_store
.Lcase7_load:
  ldaxr w0, [x1]
.Lcase7_store:
  stlxr w2, w3, [x1]
  cbnz w2, case7
  ret
.size case7, .-case7

.global dummy
.type dummy, %function
dummy:
  ret
.size dummy, .-dummy
