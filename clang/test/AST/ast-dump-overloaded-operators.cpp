// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck -strict-whitespace %s

enum E {};
void operator+(E,E);
void operator,(E,E);

void test() {
  E e;
  e + e;
  e , e;
}
// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>{{( <undeserialized declarations>)?}}
// CHECK: |-EnumDecl {{.*}} <{{.*}}ast-dump-overloaded-operators.cpp:10:1, col:9> col:6{{( imported)?}} referenced E
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:11:1, col:19> col:6{{( imported)?}} used operator+ 'void (E, E)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:16> col:17{{( imported)?}} 'E'
// CHECK:      | `-ParmVarDecl {{.*}} <col:18> col:19{{( imported)?}} 'E'
// CHECK:      |-FunctionDecl {{.*}} <line:12:1, col:19> col:6{{( imported)?}} used operator, 'void (E, E)'
// CHECK:      | |-ParmVarDecl {{.*}} <col:16> col:17{{( imported)?}} 'E'
// CHECK:      | `-ParmVarDecl {{.*}} <col:18> col:19{{( imported)?}} 'E'
// CHECK:      `-FunctionDecl {{.*}} <line:14:1, line:18:1> line:14:6{{( imported)?}} test 'void ()'
// CHECK:        `-CompoundStmt {{.*}} <col:13, line:18:1>
// CHECK:          |-DeclStmt {{.*}} <line:15:3, col:6>
// CHECK:          | `-VarDecl {{.*}} <col:3, col:5> col:5{{( imported)?}} used e 'E'
// CHECK:          |-CXXOperatorCallExpr {{.*}} <line:16:3, col:7> 'void' '+'
// CHECK:          | |-ImplicitCastExpr {{.*}} <col:5> 'void (*)(E, E)' <FunctionToPointerDecay>
// CHECK:          | | `-DeclRefExpr {{.*}} <col:5> 'void (E, E)' lvalue Function {{.*}} 'operator+' 'void (E, E)'
// CHECK:          | |-ImplicitCastExpr {{.*}} <col:3> 'E' <LValueToRValue>
// CHECK:          | | `-DeclRefExpr {{.*}} <col:3> 'E' lvalue Var {{.*}} 'e' 'E'
// CHECK:          | `-ImplicitCastExpr {{.*}} <col:7> 'E' <LValueToRValue>
// CHECK:          |   `-DeclRefExpr {{.*}} <col:7> 'E' lvalue Var {{.*}} 'e' 'E'
// CHECK:          `-CXXOperatorCallExpr {{.*}} <line:17:3, col:7> 'void' ','
// CHECK:            |-ImplicitCastExpr {{.*}} <col:5> 'void (*)(E, E)' <FunctionToPointerDecay>
// CHECK:            | `-DeclRefExpr {{.*}} <col:5> 'void (E, E)' lvalue Function {{.*}} 'operator,' 'void (E, E)'
// CHECK:            |-ImplicitCastExpr {{.*}} <col:3> 'E' <LValueToRValue>
// CHECK:            | `-DeclRefExpr {{.*}} <col:3> 'E' lvalue Var {{.*}} 'e' 'E'
// CHECK:            `-ImplicitCastExpr {{.*}} <col:7> 'E' <LValueToRValue>
// CHECK:              `-DeclRefExpr {{.*}} <col:7> 'E' lvalue Var {{.*}} 'e' 'E'
