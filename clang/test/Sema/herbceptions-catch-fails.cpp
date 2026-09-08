// RUN: not %clang_cc1 -fherbceptions -fsyntax-only %s 2>&1 | FileCheck %s

int plain(int x) { return x; }
int bar(int x) return_failure{int} { if (x < 0) return_failure x; return x + 1; }

// CHECK: error: 'catch return_failure' expression operand must be a call to a function declared 'throws' or 'return_failure{...}'
int f1() {
  auto e = catch return_failure(plain(1));
  return 0;
}
