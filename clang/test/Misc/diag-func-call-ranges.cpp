// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s --strict-whitespace

// CHECK: error: no matching function for call to 'func'

// CHECK:      :{[[@LINE+1]]:12-[[@LINE+1]]:18}: note: {{.*}} requires single argument
void func( int aa ) {}
// CHECK:      :{[[@LINE+1]]:12-[[@LINE+3]]:18}: note: {{.*}} requires 3 arguments
void func( int aa,
           int bb,
           int cc) {}

void arity_mismatch() {
  func(2, 4);
}
