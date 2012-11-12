// RUN: clang-check -ast-dump "%s" -- 2>&1 | FileCheck %s
// CHECK: namespace test_namespace
// CHECK-NEXT: class TheClass
// CHECK: int theMethod(int x) (CompoundStmt
// CHECK-NEXT:   (ReturnStmt
// CHECK-NEXT:     (BinaryOperator
//
// RUN: clang-check -ast-dump -ast-dump-filter test_namespace::TheClass::theMethod "%s" -- 2>&1 | FileCheck -check-prefix CHECK-FILTER %s
// CHECK-FILTER-NOT: namespace test_namespace
// CHECK-FILTER-NOT: class TheClass
// CHECK-FILTER: int theMethod(int x) (CompoundStmt
// CHECK-FILTER-NEXT:   (ReturnStmt
// CHECK-FILTER-NEXT:     (BinaryOperator
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
// CHECK-ATTR-NEXT: int n __attribute__((aligned((BinaryOperator
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
