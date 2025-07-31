// RUN: %clang_cc1 -std=c++20 -fsyntax-only -ast-dump %s | FileCheck %s
struct Node {
  long val;
};
template <bool>
void CallNew() {
    new Node(0);
}
// CHECK-LABEL: FunctionTemplateDecl {{.*}} CallNew
// CHECK: |-FunctionDecl {{.*}} CallNew 'void ()'
// CHECK:  `-CXXNewExpr {{.*}} 'operator new'
// CHECK:  `-CXXParenListInitExpr {{.*}} 'Node'
// CHECK:  `-ImplicitCastExpr {{.*}} 'long' <IntegralCast>
// CHECK:  `-IntegerLiteral {{.*}} 'int' 0
// CHECK: `-FunctionDecl {{.*}} used CallNew 'void ()' implicit_instantiation
// CHECK:   |-TemplateArgument integral 'true'
// CHECK:   `-CXXNewExpr {{.*}} 'operator new'
// CHECK:   `-CXXParenListInitExpr {{.*}} 'Node'
// CHECK:   `-ImplicitCastExpr {{.*}} 'long' <IntegralCast>
// CHECK:   `-IntegerLiteral {{.*}} 'int' 0
void f() {
    (void)CallNew<true>; 
}
