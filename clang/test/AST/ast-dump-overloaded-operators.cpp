// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck -strict-whitespace %s

enum E {};
void operator+(E,E);
void operator,(E,E);

void test() {
  E e;
  e + e;
  e , e;
}
// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <line:7:1, line:11:1> line:7:6 test 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:13, line:11:1>
// CHECK-NEXT:     |-DeclStmt {{.*}} <line:8:3, col:6>
// CHECK-NEXT:     | `-VarDecl {{.*}} <col:3, col:5> col:5 used e 'E'
// CHECK-NEXT:     |-CXXOperatorCallExpr {{.*}} <line:9:3, col:7> 'void' '+'
// CHECK-NEXT:     | |-ImplicitCastExpr {{.*}} <col:5> 'void (*)(E, E)' <FunctionToPointerDecay>
// CHECK-NEXT:     | | `-DeclRefExpr {{.*}} <col:5> 'void (E, E)' lvalue Function {{.*}} 'operator+' 'void (E, E)'
// CHECK-NEXT:     | |-ImplicitCastExpr {{.*}} <col:3> 'E' <LValueToRValue>
// CHECK-NEXT:     | | `-DeclRefExpr {{.*}} <col:3> 'E' lvalue Var {{.*}} 'e' 'E'
// CHECK-NEXT:     | `-ImplicitCastExpr {{.*}} <col:7> 'E' <LValueToRValue>
// CHECK-NEXT:     |   `-DeclRefExpr {{.*}} <col:7> 'E' lvalue Var {{.*}} 'e' 'E'
// CHECK-NEXT:     `-CXXOperatorCallExpr {{.*}} <line:10:3, col:7> 'void' ','
// CHECK-NEXT:       |-ImplicitCastExpr {{.*}} <col:5> 'void (*)(E, E)' <FunctionToPointerDecay>
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} <col:5> 'void (E, E)' lvalue Function {{.*}} 'operator,' 'void (E, E)'
// CHECK-NEXT:       |-ImplicitCastExpr {{.*}} <col:3> 'E' <LValueToRValue>
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} <col:3> 'E' lvalue Var {{.*}} 'e' 'E'
// CHECK-NEXT:       `-ImplicitCastExpr {{.*}} <col:7> 'E' <LValueToRValue>
// CHECK-NEXT:         `-DeclRefExpr {{.*}} <col:7> 'E' lvalue Var {{.*}} 'e' 'E'
