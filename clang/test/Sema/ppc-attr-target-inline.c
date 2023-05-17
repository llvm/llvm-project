// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le -target-feature +htm -fsyntax-only -emit-llvm %s -verify

__attribute__((always_inline))
int test1(int *x) {
  *x = __builtin_ttest();
  return *x;
}

__attribute__((target("no-htm")))
int test2(int *x) {
  *x = test1(x); // expected-error {{always_inline function 'test1' requires target feature 'htm', but would be inlined into function 'test2' that is compiled without support for 'htm'}}
  return 0;
}
