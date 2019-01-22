// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// The implicit UsingDirectiveDecls for the anonymous namespaces are created by the Sema.

// CHECK: NamespaceDecl
// The nested anonymous namespace.
// CHECK-NEXT: NamespaceDecl
// CHECK: FunctionDecl
// CHECK-SAME: func4
// CHECK-NEXT: CompoundStmt
// This is for the nested anonymous namespace.
// CHECK-NEXT: UsingDirectiveDecl
// CHECK-SAME: ''
// CHECK: FunctionDecl
// CHECK-SAME: func1
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: UsingDirectiveDecl
// CHECK-SAME: ''

// CHECK: NamespaceDecl
// CHECK-SAME: test_namespace1
// CHECK-NEXT: NamespaceDecl
// CHECK: FunctionDecl
// CHECK-SAME: func2
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: UsingDirectiveDecl
// CHECK-SAME: ''

// CHECK-NEXT: NamespaceDecl
// CHECK-SAME: test_namespace2
// CHECK-NEXT: NamespaceDecl
// CHECK-NEXT: NamespaceDecl
// CHECK-SAME: test_namespace3
// CHECK: FunctionDecl
// CHECK-SAME: func3
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: UsingDirectiveDecl
// CHECK-SAME: ''

void expr() {
  func1();
  test_namespace1::func2();
  test_namespace2::test_namespace3::func3();
  func4();
}
