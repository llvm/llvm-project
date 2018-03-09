// RUN: clang-import-test -dump-ir -use-origins -import %S/Inputs/Callee.cpp -expression %s | FileCheck %s
// CHECK: %struct.S = type { i
// CHECK: %struct.S.0 = type { i

void foo() {
  return Bar().bar(3, true);
}
