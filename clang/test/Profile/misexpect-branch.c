// Test that misexpect detects mis-annotated branches

// RUN: llvm-profdata merge %S/Inputs/misexpect-branch.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify=imprecise -Wmisexpect
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify=exact -Wmisexpect -debug-info-kind=line-tables-only
// RUN: %clang_cc1 %s -O2 -o - -disable-llvm-passes -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify=foo

// foo-no-diagnostics
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

int foo(int);
int baz(int);
int buzz();

const int inner_loop = 100;
const int outer_loop = 2000;

int bar() { // imprecise-warning-re {{Potential performance regression from use of __builtin_expect(): Annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions.}}
  int rando = buzz();
  int x = 0;
  if (likely(rando % (outer_loop * inner_loop) == 0)) { // exact-warning-re {{Potential performance regression from use of __builtin_expect(): Annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions.}}
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}
