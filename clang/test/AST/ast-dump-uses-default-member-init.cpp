// Test that a list-initialization which uses a default member initializer is
// marked as such. Within such an initializer, `this` denotes the object the
// list-initialization initializes, not the object of the enclosing member
// function.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump %s \
// RUN: | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump=json %s \
// RUN: | FileCheck --check-prefix JSON %s

struct WithDefault {
  int x;
  int *self = &x;
};

struct NoDefault {
  int x;
  int y;
};

void braces() {
  WithDefault a{1};
  NoDefault b{1, 2};
}

// CHECK-LABEL: FunctionDecl {{.*}} braces
// CHECK:         InitListExpr {{.*}} 'WithDefault' explicit uses default member init
// CHECK-NEXT:      IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:      CXXDefaultInitExpr
// The marker is absent, so the line ends after `explicit`.
// CHECK:         InitListExpr {{.*}} 'NoDefault' explicit{{$}}
// CHECK-NEXT:      IntegerLiteral {{.*}} 'int' 1

void parens() {
  WithDefault a(1);
  NoDefault b(1, 2);
}

// CHECK-LABEL: FunctionDecl {{.*}} parens
// CHECK:         CXXParenListInitExpr {{.*}} 'WithDefault' uses default member init
// CHECK-NEXT:      IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:      CXXDefaultInitExpr
// CHECK:         CXXParenListInitExpr {{.*}} 'NoDefault'{{$}}
// CHECK-NEXT:      IntegerLiteral {{.*}} 'int' 1

// JSON:      "kind": "InitListExpr",
// JSON:      "qualType": "WithDefault"
// JSON:      "usesDefaultMemberInit": true

// JSON:      "kind": "CXXParenListInitExpr",
// JSON:      "qualType": "WithDefault"
// JSON:      "usesDefaultMemberInit": true
