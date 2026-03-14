// RUN: %clang_pgogen -fprofile-function-groups=3 -fprofile-selected-function-group=0 %s -o %t.0.out
// RUN: %clang_pgogen -fprofile-function-groups=3 -fprofile-selected-function-group=1 %s -o %t.1.out
// RUN: %clang_pgogen -fprofile-function-groups=3 -fprofile-selected-function-group=2 %s -o %t.2.out
// RUN: env LLVM_PROFILE_FILE=%t.0.profraw %run %t.0.out
// RUN: env LLVM_PROFILE_FILE=%t.1.profraw %run %t.1.out
// RUN: env LLVM_PROFILE_FILE=%t.2.profraw %run %t.2.out
// RUN: llvm-profdata merge -o %t.profdata %t.*.profraw
// RUN: llvm-profdata show %t.profdata --all-functions | FileCheck %s

int foo(int i) { return 4 * i + 1; }
int bar(int i) { return 4 * i + 2; }
int goo(int i) { return 4 * i + 3; }

int main(int argc, char *argv[]) {
  foo(5);
  bar(6);
  goo(7);
  return 0;
}

// Even though we ran this code three times, we expect all counts to be one if
// functions were partitioned into groups correctly.

// CHECK: Counters: 1
// CHECK: Counters: 1
// CHECK: Counters: 1
// CHECK: Counters: 1
// CHECK: Total functions: 4
