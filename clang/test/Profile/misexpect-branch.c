// Test that misexpect detects mis-annotated branches

// test diagnostics are issued when profiling data mis-matches annotations
// RUN: llvm-profdata merge %S/Inputs/misexpect-branch.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify=imprecise -Wmisexpect
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify=exact -Wmisexpect -debug-info-kind=line-tables-only

// there should be no diagnostics when the tolerance is sufficiently high, or when -Wmisexpect is not requested
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify=foo -fdiagnostics-misexpect-tolerance=10 -Wmisexpect -debug-info-kind=line-tables-only
// RUN: %clang_cc1 %s -O2 -o - -disable-llvm-passes -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify=foo

// Ensure we emit an error when we don't use pgo with tolerance threshold
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm  -fdiagnostics-misexpect-tolerance=10 -Wmisexpect -debug-info-kind=line-tables-only 2>&1 | FileCheck -check-prefix=NO_PGO %s

// Test -fdiagnostics-misexpect-tolerance=  requires pgo profile
// NO_PGO: '-fdiagnostics-misexpect-tolerance=' requires profile-guided optimization information

// foo-no-diagnostics
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

int foo(int);
int baz(int);
int buzz();

const int inner_loop = 100;
const int outer_loop = 2000;

int bar() { // imprecise-warning-re {{potential performance regression from use of __builtin_expect(): annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions}}
  int rando = buzz();
  int x = 0;
  if (likely(rando % (outer_loop * inner_loop) == 0)) { // exact-warning-re {{potential performance regression from use of __builtin_expect(): annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions}}
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}

int fizz() { // imprecise-warning-re {{potential performance regression from use of __builtin_expect(): annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions}}
  int rando = buzz();
  int x = 0;
  if (unlikely(rando % (outer_loop * inner_loop) == 0)) { // exact-warning-re {{potential performance regression from use of __builtin_expect(): annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions}}
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}
