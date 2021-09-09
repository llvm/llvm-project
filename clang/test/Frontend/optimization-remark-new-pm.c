// Verify that remarks for the inliner appear. The remarks under the new PM will
// be slightly different than those emitted by the legacy PM. The new PM inliner
// also doesnot appear to be added at O0, so we test at O1.
// RUN: %clang_cc1 %s -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline -O1 -fexperimental-new-pass-manager -emit-llvm-only -mllvm -mandatory-inlining-first=false -verify
// RUN: %clang_cc1 %s -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline -O1 -fexperimental-new-pass-manager -emit-llvm-only -debug-info-kind=line-tables-only -mllvm -mandatory-inlining-first=false -verify

int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

float foz(int x, int y) __attribute__((noinline));
float foz(int x, int y) { return x * y; }

// The negative diagnostics are emitted twice because the inliner runs
// twice.
//
int bar(int j) {
  // expected-remark@+2 {{'foz' not inlined into 'bar' because it should never be inlined (cost=never)}}
  // expected-remark@+1 {{'foo' inlined into 'bar'}}
  return foo(j, j - 2) * foz(j - 2, j);
}
