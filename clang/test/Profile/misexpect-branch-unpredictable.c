// Test that misexpect emits no warning when prediction is correct

// RUN: llvm-profdata merge %S/Inputs/misexpect-branch.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -disable-llvm-passes -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -verify -Wmisexpect

// expected-no-diagnostics
#define unpredictable(x) __builtin_unpredictable(!!(x))

int foo(int);
int baz(int);
int buzz();

const int inner_loop = 100;
const int outer_loop = 2000;

int bar() {
  int rando = buzz();
  int x = 0;
  if (unpredictable(rando % (outer_loop * inner_loop) == 0)) {
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}
