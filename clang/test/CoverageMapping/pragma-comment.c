// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

// Comments in _Pragma strings are lexed in macro expansions.

void f(void) {
  _Pragma("clang diagnostic push /* comment */")
}

// CHECK-LABEL: f:
// CHECK-NEXT: File 0, [[@LINE-5]]:14 -> [[@LINE-3]]:2 = #0
// CHECK-NOT: Skipped
