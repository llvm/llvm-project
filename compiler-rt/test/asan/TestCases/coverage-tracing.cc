// Test -fsanitize-coverage=trace-bb
//
// RUN: %clangxx_asan -O1 -fsanitize-coverage=func,trace-bb %s -o %t
// RUN: rm -rf   %T/coverage-tracing
// RUN: mkdir %T/coverage-tracing
// RUN: cd %T/coverage-tracing
// RUN:  A=x;   %env_asan_opts=coverage=1:verbosity=1 %run %t $A 1   2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK1; mv trace-points.*.sancov $A.points
// RUN:  A=f;   %env_asan_opts=coverage=1:verbosity=1 %run %t $A 1   2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK2; mv trace-points.*.sancov $A.points
// RUN:  A=b;   %env_asan_opts=coverage=1:verbosity=1 %run %t $A 1   2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK2; mv trace-points.*.sancov $A.points
// RUN:  A=bf;  %env_asan_opts=coverage=1:verbosity=1 %run %t $A 1   2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK3; mv trace-points.*.sancov $A.points
// RUN:  A=fb;  %env_asan_opts=coverage=1:verbosity=1 %run %t $A 1   2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK3; mv trace-points.*.sancov $A.points
// RUN:  A=ffb; %env_asan_opts=coverage=1:verbosity=1 %run %t $A 1   2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK4; mv trace-points.*.sancov $A.points
// RUN:  A=fff; %env_asan_opts=coverage=1:verbosity=1 %run %t $A 1   2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK4; mv trace-points.*.sancov $A.points
// RUN:  A=bbf; %env_asan_opts=coverage=1:verbosity=1 %run %t $A 100 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK301; mv trace-points.*.sancov $A.points
// RUN: diff f.points fff.points
// RUN: diff bf.points fb.points
// RUN: diff bf.points ffb.points
// RUN: diff bf.points bbf.points
// RUN: not diff x.points f.points
// RUN: not diff x.points b.points
// RUN: not diff x.points bf.points
// RUN: not diff f.points b.points
// RUN: not diff f.points bf.points
// RUN: not diff b.points bf.points
// RUN: rm -rf   %T/coverage-tracing
//
// REQUIRES: asan-64-bits, shell
// UNSUPPORTED: android

#include <stdlib.h>
volatile int sink;
__attribute__((noinline)) void foo() { sink++; }
__attribute__((noinline)) void bar() { sink++; }

int main(int argc, char **argv) {
  if (argc != 3) return 0;
  int n = strtol(argv[2], 0, 10);
  while (n-- > 0) {
    for (int i = 0; argv[1][i]; i++) {
      if (argv[1][i] == 'f') foo();
      else if (argv[1][i] == 'b') bar();
    }
  }
}

// CHECK: CovDump: Trace: 3 PCs written
// CHECK1: CovDump: Trace: 1 Events written
// CHECK2: CovDump: Trace: 2 Events written
// CHECK3: CovDump: Trace: 3 Events written
// CHECK4: CovDump: Trace: 4 Events written
// CHECK301: CovDump: Trace: 301 Events written
