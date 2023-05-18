// This test checks that we are no instrumenting a memory access twice
// (before and after inlining)
// RUN: %clangxx_asan -O1 %s -o %t && %run %t
// RUN: %clangxx_asan -O0 %s -o %t && %run %t
__attribute__((always_inline))
void foo(int *x) {
  *x = 0;
}

int main() {
  int x;
  foo(&x);
  return x;
}
