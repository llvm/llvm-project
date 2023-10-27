// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s --strict-whitespace

// CHECK:      :{9:3-9:7}: error: too few arguments
// CHECK:      :{7:12-7:26}: note: 'func' declared here
// CHECK:      :{10:3-10:7}{10:13-10:17}: error: too many arguments
// CHECK:      :{7:12-7:26}: note: 'func' declared here
void func( int aa, int bb) {}
void arity_mismatch() {
  func(3);
  func(3, 4,5, 6);
}
