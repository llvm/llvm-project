// RUN: %clang_pgogen -mllvm -pgo-function-entry-coverage %s -o %t.out
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.out
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --covered %t.profdata | FileCheck %s --implicit-check-not goo

// We deliberately merge the raw profile twice to test that internal counts can
// grow larger than one. Technically, accumulating coverage values is different
// than accumulating counts, but this helps discriminate cold functions from hot
// functions when the number of raw profiles is large.
// RUN: llvm-profdata merge -o %t2.profdata %t.profraw %t.profraw
// RUN: llvm-profdata show %t2.profdata | FileCheck %s --check-prefix=COUNTS

// RUN: %clang_cspgogen -O1 -mllvm -pgo-function-entry-coverage %s -o %t.cs.out
// RUN: env LLVM_PROFILE_FILE=%t.csprofraw %run %t.cs.out
// RUN: llvm-profdata merge -o %t.csprofdata %t.csprofraw
// RUN: llvm-profdata show --covered %t.csprofdata --showcs | FileCheck %s --implicit-check-not goo
// RUN: llvm-profdata merge -o %t2.csprofdata %t.csprofraw %t.csprofraw
// RUN: llvm-profdata show --showcs %t2.csprofdata | FileCheck %s --check-prefix=COUNTS

void markUsed(int a) {
  volatile int g;
  g = a;
}

__attribute__((noinline)) int foo(int i) { return 4 * i + 1; }
__attribute__((noinline)) int bar(int i) { return 4 * i + 2; }
__attribute__((noinline)) int goo(int i) { return 4 * i + 3; }

int main(int argc, char *argv[]) {
  markUsed(foo(5));
  markUsed(argc ? bar(6) : goo(7));
  return 0;
}

// CHECK-DAG: main
// CHECK-DAG: foo
// CHECK-DAG: bar

// COUNTS: Maximum function count: 2
