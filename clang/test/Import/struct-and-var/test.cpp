// RUN: clang-import-test -dump-ast --import %S/Inputs/S1.cpp --import %S/Inputs/S2.cpp -expression %s | FileCheck %s

// CHECK: {{^`-CXXRecordDecl}}
// CHECK-SAME: Inputs/S2.cpp:1:1, line:3:1> line:1:8 struct F

void expr() {
  struct F f;
  int x = f.a;
}
