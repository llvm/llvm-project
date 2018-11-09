// RUN: clang-check -ast-dump "%s" -- 2>&1 | FileCheck %s
// CHECK: NamespaceDecl{{.*}}test_namespace
// CHECK-NEXT: CXXRecordDecl{{.*}}TheClass
// CHECK: CXXMethodDecl{{.*}}theMethod
// CHECK-NEXT: ParmVarDecl{{.*}}x
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT:   ReturnStmt
// CHECK-NEXT:     BinaryOperator
//
// RUN: clang-check -ast-dump -ast-dump-filter test_namespace::TheClass::theMethod "%s" -- 2>&1 | FileCheck -check-prefix CHECK-FILTER %s
// CHECK-FILTER-NOT: NamespaceDecl
// CHECK-FILTER-NOT: CXXRecordDecl
// CHECK-FILTER: {{^}}Dumping test_namespace::TheClass::theMethod
// CHECK-FILTER-NEXT: {{^}}CXXMethodDecl{{.*}}theMethod
// CHECK-FILTER-NEXT: ParmVarDecl{{.*}}x
// CHECK-FILTER-NEXT: CompoundStmt
// CHECK-FILTER-NEXT:   ReturnStmt
// CHECK-FILTER-NEXT:     BinaryOperator
//
// RUN: clang-check -ast-print "%s" -- 2>&1 | FileCheck -check-prefix CHECK-PRINT %s
// CHECK-PRINT: namespace test_namespace
// CHECK-PRINT: class TheClass
// CHECK-PRINT: int theMethod(int x)
//
// RUN: clang-check -ast-list "%s" -- 2>&1 | FileCheck -check-prefix CHECK-LIST %s
// CHECK-LIST: test_namespace
// CHECK-LIST-NEXT: test_namespace::TheClass
// CHECK-LIST-NEXT: test_namespace::TheClass::theMethod
// CHECK-LIST-NEXT: x
//
// RUN: clang-check -ast-dump -ast-dump-filter test_namespace::TheClass::n "%s" -- 2>&1 | FileCheck -check-prefix CHECK-ATTR %s
// CHECK-ATTR: test_namespace
// CHECK-ATTR-NEXT: FieldDecl{{.*}}n
// CHECK-ATTR-NEXT:   AlignedAttr
// CHECK-ATTR-NEXT:     ConstantExpr
// CHECK-ATTR-NEXT:       BinaryOperator
//
// RUN: clang-check -ast-dump -ast-dump-filter test_namespace::AfterNullNode "%s" -- 2>&1 | FileCheck -check-prefix CHECK-AFTER-NULL %s
// CHECK-AFTER-NULL: class AfterNullNode

namespace test_namespace {

class TheClass {
public:
  int theMethod(int x) {
    return x + x;
  }
  int n __attribute__((aligned(1+1)));
};

// Used to fail with -ast-dump-filter X
template<template<typename T> class C> class Z {};

// Check that traversal continues after the previous construct.
class AfterNullNode {};

}
