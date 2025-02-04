// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s

struct S {
  struct {
    int i;
  };
};

int accessInRegularFunction() {
  return S().i;
  // CHECK: FunctionDecl {{.*}} accessInRegularFunction 'int ()'
  // CHECK:      |   `-ReturnStmt {{.*}}
  // CHECK-NEXT: |     `-ExprWithCleanups {{.*}} 'int'
  // CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: |         `-MemberExpr {{.*}} 'int' xvalue .i
  // CHECK-NEXT: |           `-MemberExpr {{.*}} 'S::(anonymous struct at {{.*}})
  // CHECK-NEXT: |             `-MaterializeTemporaryExpr {{.*}} 'S' xvalue
  // CHECK-NEXT: |               `-CXXTemporaryObjectExpr {{.*}} 'S' 'void () noexcept' zeroing
}

// AST should look the same in a function template with an unused template
// parameter.
template <class>
int accessInFunctionTemplate() {
  return S().i;
  // CHECK: FunctionDecl {{.*}} accessInFunctionTemplate 'int ()'
  // CHECK:      |   `-ReturnStmt {{.*}}
  // CHECK-NEXT: |     `-ExprWithCleanups {{.*}} 'int'
  // CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: |         `-MemberExpr {{.*}} 'int' xvalue .i
  // CHECK-NEXT: |           `-MemberExpr {{.*}} 'S::(anonymous struct at {{.*}})
  // CHECK-NEXT: |             `-MaterializeTemporaryExpr {{.*}} 'S' xvalue
  // CHECK-NEXT: |               `-CXXTemporaryObjectExpr {{.*}} 'S' 'void () noexcept' zeroing
}

// AST should look the same in an instantiation of the function template.
// This is a regression test: The AST used to contain the
// `MaterializeTemporaryExpr` in the wrong place, causing a `MemberExpr` to have
// a prvalue base (which is not allowed in C++).
template int accessInFunctionTemplate<int>();
  // CHECK: FunctionDecl {{.*}} accessInFunctionTemplate 'int ()' explicit_instantiation_definition
  // CHECK:          `-ReturnStmt {{.*}}
  // CHECK-NEXT:       `-ExprWithCleanups {{.*}} 'int'
  // CHECK-NEXT:         `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT:           `-MemberExpr {{.*}} 'int' xvalue .i
  // CHECK-NEXT:             `-MemberExpr {{.*}} 'S::(anonymous struct at {{.*}})
  // CHECK-NEXT:               `-MaterializeTemporaryExpr {{.*}} 'S' xvalue
  // CHECK-NEXT:                 `-CXXTemporaryObjectExpr {{.*}} 'S' 'void () noexcept' zeroing
