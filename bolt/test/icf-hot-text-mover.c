// Check that ICF does not fold hot text mover functions.
// Hot text mover functions are placed in special sections (e.g., .never_hugify)
// to avoid being placed on hot/huge pages.

// clang-format off

// REQUIRES: system-linux, asserts
// RUN: %clang %cflags -O0 %s -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe --icf --hot-text-move-sections=.never_hugify \
// RUN:   -debug-only=bolt-icf -o %t.bolt 2>&1 | FileCheck %s

// FuncA is in .text, FuncB is in .never_hugify (a hot text mover section).
// They are identical, but FuncB should NOT be folded because it is a hot text mover.
// CHECK: ICF iteration 1
// CHECK-NOT: folding FuncB into FuncA
// CHECK-NOT: folding FuncA into FuncB

__attribute__((noinline)) int FuncA(void) { return 42; }

__attribute__((section(".never_hugify"), noinline)) int FuncB(void) {
  return 42;
}

int main(void) { return FuncA() + FuncB(); }
