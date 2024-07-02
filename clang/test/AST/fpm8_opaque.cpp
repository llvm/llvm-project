// RUN: %clang_cc1 -std=c++11 -ast-dump %s | FileCheck %s --strict-whitespace

/*  Various contexts where type __fpm8 can appear. */

/*  Namespace */
namespace {
  __fpm8 f2n;
  __fpm8 arr1n[10];
}

//CHECK:       |-NamespaceDecl {{.*}}
//CHECK-NEXT:  | |-VarDecl {{.*}} f2n '__fpm8'
//CHECK-NEXT:  | `-VarDecl {{.*}} arr1n '__fpm8[10]'

  __fpm8 arr1[10];
  //__fpm8 arr2n[] { 1, 3, 3 }; cannot initialize
  
  const __fpm8 func1n(const __fpm8 fpm8) {
    // this should fail
    __fpm8 f1n;
    f1n  = fpm8;
    return f1n;
  }

//CHECK:        |-VarDecl {{.*}} '__fpm8[10]'

//CHECK:            | `-VarDecl {{.*}} f1n '__fpm8'
//CHECK-NEXT:       |-BinaryOperator {{.*}} '__fpm8' lvalue '='
//CHECK-NEXT:       | |-DeclRefExpr {{.*}} '__fpm8' lvalue Var {{.*}} 'f1n' '__fpm8'
//CHECK-NEXT:       | `-ImplicitCastExpr {{.*}} '__fpm8' <LValueToRValue>
//CHECK-NEXT:       |   `-DeclRefExpr {{.*}} 'const __fpm8' lvalue ParmVar {{.*}} 'fpm8' 'const __fpm8'
//CHECK-NEXT:        `-ReturnStmt {{.*}}
//CHECK-NEXT:         `-ImplicitCastExpr {{.*}} '__fpm8' <LValueToRValue>
//CHECK-NEXT:           `-DeclRefExpr {{.*}} '__fpm8' lvalue Var {{.*}} 'f1n' '__fpm8'


/* Class */

class C1 {
  __fpm8 f1c;
  static const __fpm8 f2c;
  volatile __fpm8 f3c;
public:
  C1(__fpm8 arg) : f1c(arg), f3c(arg) { }
  __fpm8 func1c(__fpm8 arg ) {
    return  arg;
  }
  static __fpm8 func2c(__fpm8 arg) {
    return arg;
  }
};

//CHECK:       | |-CXXRecordDecl {{.*}} referenced class C1
//CHECK-NEXT:  | |-FieldDecl {{.*}} f1c '__fpm8'
//CHECK-NEXT:  | |-VarDecl {{.*}} f2c 'const __fpm8' static
//CHECK-NEXT:  | |-FieldDecl {{.*}} f3c 'volatile __fpm8'
//CHECK-NEXT:  | |-AccessSpecDecl {{.*}}
//CHECK-NEXT:  | |-CXXConstructorDecl {{.*}} C1 'void (__fpm8)' implicit-inline
//CHECK-NEXT:  | | |-ParmVarDecl {{.*}} arg '__fpm8'
//CHECK-NEXT:  | | |-CXXCtorInitializer {{.*}} 'f1c' '__fpm8'
//CHECK-NEXT:  | | | `-ImplicitCastExpr {{.*}} '__fpm8' <LValueToRValue>
//CHECK-NEXT:  | | |   `-DeclRefExpr {{.*}} '__fpm8' lvalue ParmVar {{.*}} 'arg' '__fpm8'
//CHECK-NEXT:  | | |-CXXCtorInitializer {{.*}} 'f3c' 'volatile __fpm8'
//CHECK-NEXT:  | | | `-ImplicitCastExpr {{.*}} '__fpm8' <LValueToRValue>
//CHECK-NEXT:  | | |   `-DeclRefExpr {{.*}} '__fpm8' lvalue ParmVar {{.*}} 'arg' '__fpm8'
//CHECK-NEXT:  | | `-CompoundStmt {{.*}}
//CHECK-NEXT:  | |-CXXMethodDecl {{.*}} func1c '__fpm8 (__fpm8)' implicit-inline
//CHECK-NEXT:  | | |-ParmVarDecl {{.*}} arg '__fpm8'
//CHECK-NEXT:  | | `-CompoundStmt {{.*}}
//CHECK-NEXT:  | |   `-ReturnStmt {{.*}}
//CHECK-NEXT:  | |     `-ImplicitCastExpr {{.*}} '__fpm8' <LValueToRValue>
//CHECK-NEXT:  | |       `-DeclRefExpr {{.*}} '__fpm8' lvalue ParmVar {{.*}}8 'arg' '__fpm8'
//CHECK-NEXT:  | `-CXXMethodDecl {{.*}} func2c '__fpm8 (__fpm8)' static implicit-inline
//CHECK-NEXT:  |   |-ParmVarDecl {{.*}} arg '__fpm8'
//CHECK-NEXT:  |   `-CompoundStmt {{.*}}
//CHECK-NEXT:  |     `-ReturnStmt {{.*}}
//CHECK-NEXT:  |       `-ImplicitCastExpr {{.*}} '__fpm8' <LValueToRValue>
//CHECK-NEXT:  |         `-DeclRefExpr {{.*}} '__fpm8' lvalue ParmVar {{.*}} 'arg' '__fpm8'

template <class C> struct S1 {
  C mem1;
};

template <> struct S1<__fpm8> {
  __fpm8 mem2;
};

//CHECK:       |-TemplateArgument type '__fpm8'
//CHECK-NEXT:  | `-BuiltinType {{.*}} '__fpm8'
//CHECK-NEXT:  |-CXXRecordDecl {{.*}} implicit struct S1
//CHECK-NEXT:  `-FieldDecl {{.*}} mem2 '__fpm8'
