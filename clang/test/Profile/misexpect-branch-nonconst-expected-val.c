// Test that misexpect emits no warning when condition is not a compile-time constant

// RUN: llvm-profdata merge %S/Inputs/misexpect-branch-nonconst-expect-arg.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -disable-llvm-passes -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -verify -Wmisexpect

// expected-no-diagnostics
int foo(int);
int baz(int);
int buzz();

const int inner_loop = 100;
const int outer_loop = 2000;

int bar() {
  int rando = buzz();
  int x = 0;
  if (__builtin_expect(rando % (outer_loop * inner_loop) == 0, buzz())) {
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}
