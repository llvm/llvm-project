/// Test that missing annotation diagnostics are suggested for hot branches

// note test diagnostics are issued when profiling data mis-matches annotations
// RUN: llvm-profdata merge %S/Inputs/missing-annotation.proftext -o %t.profdata
// RUN: %clang %s -O2 -c -S -emit-llvm -o - -fprofile-instr-use=%t.profdata -Xclang -verify=exact -fdiagnostics-missing-annotations -debug-info-kind=line-tables-only -Rpass=missing-annotations

// foo-no-diagnostics

int foo(int);
int baz(int);
int buzz();

const int inner_loop = 100;
const int outer_loop = 2000;
int bar() { 
  int rando = buzz();
  int x = 0;
  if (rando % (outer_loop * inner_loop) == 0) { // exact-remark {{Extremely hot condition. Consider adding llvm.expect intrinsic}}
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}

int fizz() {
  int rando = buzz();
  int x = 0;
  if (rando % (outer_loop * inner_loop) == 0) { // exact-remark {{Extremely hot condition. Consider adding llvm.expect intrinsic}}
    x = baz(rando);
  } else {
    x = foo(50);
  }
  return x;
}
