// Based on lld/test/ELF/start-lib.s

// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown \
// RUN:   %p/Inputs/start-lib1.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown \
// RUN:   %p/Inputs/start-lib2.s -o %t3.o

// RUN: wasm-ld --no-gc-sections -o %t3 %t1.o %t2.o %t3.o
// RUN: obj2yaml %t3 | FileCheck --check-prefix=TEST1 %s
// TEST1: Name: foo
// TEST1: Name: bar

// RUN: wasm-ld --no-gc-sections -o %t3 %t1.o -u bar --start-lib %t2.o %t3.o
// RUN: obj2yaml %t3 | FileCheck --check-prefix=TEST2 %s
// TEST2-NOT: Name: foo
// TEST2: Name: bar

// RUN: wasm-ld --no-gc-sections -o %t3 %t1.o --start-lib %t2.o %t3.o
// RUN: obj2yaml %t3 | FileCheck --check-prefix=TEST3 %s
// TEST3-NOT: Name: foo
// TEST3-NOT: Name: bar

// RUN: not wasm-ld %t1.o --start-lib --start-lib 2>&1 | FileCheck -check-prefix=NESTED-LIB %s
// NESTED-LIB: nested --start-lib

// RUN: not wasm-ld --end-lib 2>&1 | FileCheck -check-prefix=END %s
// END: stray --end-lib

.globl _start
_start:
  .functype _start () -> ()
  end_function
