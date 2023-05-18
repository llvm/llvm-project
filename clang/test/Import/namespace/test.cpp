// RUN: clang-import-test -dump-ast -import %S/Inputs/NS.cpp -expression %s | FileCheck %s

// CHECK: `-NamespaceDecl
// CHECK-SAME: Inputs/NS.cpp:1:1, line:5:1> line:1:11 NS

void expr() {
  static_assert(NS::A == 3);
}
