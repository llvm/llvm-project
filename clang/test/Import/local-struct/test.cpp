// RUN: clang-import-test -dump-ir -import %S/Inputs/Callee.cpp -expression %s | FileCheck %s
// XFAIL: *
// CHECK: %struct.S = type { i
// CHECK: %struct.S.0 = type { i1 }

void foo() {
  return Bar().bar(3, true);
}
