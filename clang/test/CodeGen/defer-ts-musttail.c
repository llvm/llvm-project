// RUN: %clang_cc1 -triple x86_64-unknown-linux -std=c23 -fdefer-ts -emit-llvm %s -o /dev/null -verify

int bar() { return 12; }
int foo() {
  _Defer {};
  [[clang::musttail]] return bar(); // expected-error {{cannot compile this tail call skipping over cleanups yet}}
}
