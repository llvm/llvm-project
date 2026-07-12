// Exercise compact-unwind through JIT-linked frames under MachOPlatform.
//
// Each object gets its own per-graph compact-unwind base (a local Mach-O
// header) rather than sharing the JITDylib header, so unwinding works no
// matter where in the address space a graph is emitted -- including below the
// header. A throw that crosses a separately-linked object's frame forces the
// unwinder to decode __unwind_info for a graph other than the one holding the
// header.
//
// RUN: %clangxx -fexceptions -fPIC -emit-llvm -c -o %t.throw.bc %s
// RUN: %clangxx -DMAIN -fexceptions -fPIC -emit-llvm -c -o %t.main.bc %s
// RUN: %lli_orc_jitlink -relocation-model=pic -extra-module %t.throw.bc \
// RUN:     %t.main.bc | FileCheck %s

// CHECK: in throw_it
// CHECK-NEXT: caught 42

#include <stdio.h>

#ifdef MAIN

void throw_it();

int main() {
  try {
    throw_it();
  } catch (int X) {
    printf("caught %d\n", X);
  }
  return 0;
}

#else

void throw_it() {
  puts("in throw_it");
  throw 42;
}

#endif
