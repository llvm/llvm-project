// RUN: clang-import-test -dump-ir -import %S/Inputs/Callee.cpp -expression %s | FileCheck %s
// CHECK: %struct.S = type { i

void foo() {
  return Bar().bar(3);
}
