// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// The implicit UsingDirectiveDecls for the anonymous namespaces are created by the Sema.

// There might be another builtin namespace before our first namespace, so we can't
// just look for NamespaceDecl. Instead look for the first line of F.cpp (which only
// contains the namespace we are looking for but no other decl).
// CHECK: F.cpp:1:1
// The nested anonymous namespace.
// CHECK-NEXT: NamespaceDecl
// CHECK-SAME: line:21:11
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
