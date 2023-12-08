// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s
// CHECK: CXXCtorInitializer
// CHECK-SAME: 'int_member'
// CHECK-SAME: 'int'
// CHECK-NEXT: CXXDefaultInitExpr
// CHECK-SAME: 'int'

// CHECK-NEXT: CXXCtorInitializer
// CHECK-SAME: 'float_member'
// CHECK-SAME: 'float'
// CHECK-NEXT: CXXDefaultInitExpr
// CHECK-SAME: 'float'

// CHECK-NEXT: CXXCtorInitializer
// CHECK-SAME: 'class_member'
// CHECK-SAME: 'Foo'
// CHECK-NEXT: CXXDefaultInitExpr
// CHECK-SAME: 'Foo'

void expr() {
  struct S s;
}
