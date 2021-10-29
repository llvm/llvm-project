// RUN: not %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fms-extensions -ast-dump -ast-dump-filter Test %s | FileCheck -check-prefix CHECK -strict-whitespace %s

namespace TestInvalidRParenOnCXXUnresolvedConstructExpr {
template <class T>
void f(T i, T j) {
  return T (i, j;
}
}

// CHECK: NamespaceDecl {{.*}} <{{.*}}> {{.*}} TestInvalidRParenOnCXXUnresolvedConstructExpr
// CHECK-NEXT: `-FunctionTemplateDecl
// CHECK-NEXT:   |-TemplateTypeParmDecl
// CHECK-NEXT:   `-FunctionDecl
// CHECK-NEXT:     |-ParmVarDecl
// CHECK-NEXT:     |-ParmVarDecl
// CHECK-NEXT:     `-CompoundStmt
// CHECK-NEXT:       `-ReturnStmt
// CHECK-NEXT:         `-CXXUnresolvedConstructExpr {{.*}} <col:10, col:16> 'T'
// CHECK-NEXT:           |-DeclRefExpr {{.*}} <col:13> 'T' lvalue ParmVar {{.*}} 'i' 'T'
// CHECK-NEXT:           `-DeclRefExpr {{.*}} <col:16> 'T' lvalue ParmVar {{.*}} 'j' 'T'


namespace TestInvalidIf {
int g(int i) {
  if (invalid_condition)
    return 4;
  else
    return i;
}
}
// CHECK: NamespaceDecl {{.*}} <{{.*}}> {{.*}} TestInvalidIf
// CHECK-NEXT: `-FunctionDecl
// CHECK-NEXT:   |-ParmVarDecl
// CHECK-NEXT:   `-CompoundStmt
// CHECK-NEXT:     `-IfStmt {{.*}} <line:25:3, line:28:12>
// CHECK-NEXT:       |-OpaqueValueExpr {{.*}} <<invalid sloc>> 'bool'
// CHECK-NEXT:       |-ReturnStmt {{.*}} <line:26:5, col:12>
// CHECK-NEXT:       | `-IntegerLiteral {{.*}} <col:12> 'int' 4
// CHECK-NEXT:       `-ReturnStmt {{.*}} <line:28:5, col:12>
// CHECK-NEXT:         `-ImplicitCastExpr {{.*}} <col:12> 'int' <LValueToRValue>
// CHECK-NEXT:           `-DeclRefExpr {{.*}} <col:12> 'int' lvalue ParmVar {{.*}} 'i' 'int'

namespace TestInvalidFunctionDecl {
struct Str {
   double foo1(double, invalid_type);
};
double Str::foo1(double, invalid_type)
{ return 45; }
}
// CHECK: NamespaceDecl {{.*}} <{{.*}}> {{.*}} TestInvalidFunctionDecl
// CHECK-NEXT: |-CXXRecordDecl {{.*}} <line:44:1, line:46:1> line:44:8 struct Str definition
// CHECK:      | |-CXXRecordDecl {{.*}} <col:1, col:8> col:8 implicit struct Str
// CHECK-NEXT: | `-CXXMethodDecl {{.*}} <line:45:4, col:36> col:11 invalid foo1 'double (double, int)'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} <col:16> col:22 'double'
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} <col:24, <invalid sloc>> col:36 invalid 'int'
// CHECK-NEXT: `-CXXMethodDecl {{.*}} parent {{.*}} <line:47:1, line:48:14> line:47:13 invalid foo1 'double (double, int)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:18> col:24 'double'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:26, <invalid sloc>> col:38 invalid 'int'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <line:48:1, col:14>
// CHECK-NEXT:     `-ReturnStmt {{.*}} <col:3, col:10>
// CHECK-NEXT:       `-ImplicitCastExpr {{.*}} <col:10> 'double' <IntegralToFloating>
// CHECK-NEXT:         `-IntegerLiteral {{.*}} <col:10> 'int' 45
