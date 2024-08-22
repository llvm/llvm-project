// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

/// Test that missing-annotations detects branches that are hot, but not annotated.
// RUN: llvm-profdata merge %t/a.proftext -o %t/profdata
// RUN: %clang_cc1 %t/a.c -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t/profdata -verify -mllvm -pgo-missing-annotations -Rpass=missing-annotations  -fdiagnostics-misexpect-tolerance=10

/// Test that we don't report any diagnostics, if the threshold isn't exceeded.
// RUN: %clang_cc1 %t/a.c -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t/profdata -mllvm -pgo-missing-annotations -Rpass=missing-annotations  2>&1 | FileCheck -implicit-check-not=remark %s

//--- a.c
// foo-no-diagnostics
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

int foo(int);
int baz(int);
int buzz(void);

const int inner_loop = 100;
const int outer_loop = 2000;

int bar(void) { //  imprecise-remark-re {{Extremely hot condition. Consider adding llvm.expect intrinsic{{.*}}}}
  int a = buzz();
  int x = 0;
  if (a % (outer_loop * inner_loop) == 0) { // expected-remark {{Extremely hot condition. Consider adding llvm.expect intrinsic}}
    x = baz(a);
  } else {
    x = foo(50);
  }
  return x;
}

int fizz(void) {
  int a = buzz();
  int x = 0;
  if ((a % (outer_loop * inner_loop) == 0)) { // expected-remark-re {{Extremely hot condition. Consider adding llvm.expect intrinsic{{.*}}}}}
    x = baz(a);
  } else {
    x = foo(50);
  }
  return x;
}

//--- a.proftext
bar
# Func Hash:
11262309464
# Num Counters:
2
# Counter Values:
1901
99

fizz
# Func Hash:
11262309464
# Num Counters:
2
# Counter Values:
1901
99

