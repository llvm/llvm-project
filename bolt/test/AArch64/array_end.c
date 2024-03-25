// Test checks that bolt properly finds end section label.
// Linker script contains gap after destructor array, so
// __init_array_end address would not be owned by any section.

// REQUIRES: system-linux
// RUN: %clang %cflags -no-pie %s -o %t.exe -Wl,-q \
// RUN:   -Wl,-T %S/Inputs/array_end.lld_script
// RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm \
// RUN:  --print-only="callFini" | FileCheck %s

// CHECK: adr [[REG:x[0-28]+]], "__fini_array_end/1"

__attribute__((destructor)) void destr() {}

__attribute__((noinline)) void callFini() {
  extern void (*__fini_array_start[])();
  extern void (*__fini_array_end[])();
  unsigned long Count = __fini_array_end - __fini_array_start;
  for (unsigned long I = 0; I < Count; ++I)
    (*__fini_array_start[I])();
}

void _start() { callFini(); }
