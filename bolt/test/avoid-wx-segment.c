// Test bolt instrumentation won't generate a binary with any segment that
// is writable and executable. Basically we want to put `.bolt.instr.counters`
// section into its own segment, separated from its surrounding RX sections.

// REQUIRES: system-linux

void foo() {}
void bar() { foo(); }

// RUN: %clang %cflags -c %s -o %t.o
// RUN: ld.lld -q -o %t.so %t.o -shared --init=foo --fini=foo
// RUN: llvm-bolt --instrument %t.so -o %tt.so
// RUN: llvm-readelf -l %tt.so | FileCheck %s
// CHECK-NOT: RWE
// CHECK: {{[0-9]*}} .bolt.instr.counters {{$}}
