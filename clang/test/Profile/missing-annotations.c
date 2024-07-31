// Test that missing-annotations detects branches that are hot, but not annotated

// test diagnostics are issued when profiling data mis-matches annotations
// RUN: llvm-profdata merge %S/Inputs/missing-annotations-branch.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify -mllvm -pgo-missing-annotations -Rpass=missing-annotations

// Ensure we emit an error when we don't use pgo with tolerance threshold
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm  -fdiagnostics-misexpect-tolerance=10 -mllvm -pgo-missing-annotations -debug-info-kind=line-tables-only 2>&1 | FileCheck -check-prefix=NOPGO %s

// Test -fdiagnostics-misexpect-tolerance=  requires pgo profile
// NOPGO: '-fdiagnostics-misexpect-tolerance=' requires profile-guided optimization information

// foo-no-diagnostics
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

int foo(int);
int baz(int);
int buzz(void);

const int inner_loop = 100;
const int outer_loop = 2000;

int bar(void) { //  imprecise-remark-re {{Extremely hot condition. Consider adding llvm.expect intrinsic{{.*}}}}

  int rando = buzz();
  int x = 0;
  if (rando % (outer_loop * inner_loop) == 0) { // expected-remark {{Extremely hot condition. Consider adding llvm.expect intrinsic}}
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}

int fizz(void) { // 
  int rando = buzz();
  int x = 0;
  if ((rando % (outer_loop * inner_loop) == 0)) { // expected-remark-re {{Extremely hot condition. Consider adding llvm.expect intrinsic{{.*}}}}}
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}
