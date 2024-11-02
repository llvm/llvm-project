// XFAIL: aix
// RUN: %clang_pgogen -mllvm -pgo-function-entry-coverage %s -o %t.out
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.out
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --covered %t.profdata | FileCheck %s --check-prefix CHECK --implicit-check-not goo

int foo(int i) { return 4 * i + 1; }
int bar(int i) { return 4 * i + 2; }
int goo(int i) { return 4 * i + 3; }

int main(int argc, char *argv[]) {
  foo(5);
  argc ? bar(6) : goo(7);
  return 0;
}

// CHECK: main
// CHECK: foo
// CHECK: bar
