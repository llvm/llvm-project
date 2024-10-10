// RUN: %clang_cc1 -std=c++11 -triple aarch64-arm-none-eabi -target-feature -fp8 -ast-dump %s | \
// RUN:  FileCheck %s --strict-whitespace

// REQUIRES: aarch64-registered-target || arm-registered-target

/*  Various contexts where type __mfp8 can appear. */

/*  Namespace */
namespace {
  __mfp8 f2n;
  __mfp8 arr1n[10];
}

//CHECK:       |-NamespaceDecl {{.*}}
//CHECK-NEXT:  | |-VarDecl {{.*}} f2n '__MFloat8_t'
//CHECK-NEXT:  | `-VarDecl {{.*}} arr1n '__MFloat8_t[10]'

  __mfp8 arr1[10];
  //__mfp8 arr2n[] { 1, 3, 3 }; cannot initialize
  
  const __mfp8 func1n(const __mfp8 mfp8) {
    // this should fail
    __mfp8 f1n;
    f1n  = mfp8;
    return f1n;
  }

//CHECK:        |-VarDecl {{.*}} '__MFloat8_t[10]'

//CHECK:            | `-VarDecl {{.*}} f1n '__MFloat8_t'
//CHECK-NEXT:       |-BinaryOperator {{.*}} '__MFloat8_t' lvalue '='
//CHECK-NEXT:       | |-DeclRefExpr {{.*}} '__MFloat8_t' lvalue Var {{.*}} 'f1n' '__MFloat8_t'
//CHECK-NEXT:       | `-ImplicitCastExpr {{.*}} '__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:       |   `-DeclRefExpr {{.*}} 'const __MFloat8_t' lvalue ParmVar {{.*}} 'mfp8' 'const __MFloat8_t'
//CHECK-NEXT:        `-ReturnStmt {{.*}}
//CHECK-NEXT:         `-ImplicitCastExpr {{.*}} '__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:           `-DeclRefExpr {{.*}} '__MFloat8_t' lvalue Var {{.*}} 'f1n' '__MFloat8_t'


/* Class */

class C1 {
  __mfp8 f1c;
  static const __mfp8 f2c;
  volatile __MFloat8_t f3c;
public:
  C1(__mfp8 arg) : f1c(arg), f3c(arg) { }
  __mfp8 func1c(__mfp8 arg ) {
    return  arg;
  }
  static __mfp8 func2c(__mfp8 arg) {
    return arg;
  }
};

//CHECK:       | |-CXXRecordDecl {{.*}} referenced class C1
//CHECK-NEXT:  | |-FieldDecl {{.*}} f1c '__MFloat8_t'
//CHECK-NEXT:  | |-VarDecl {{.*}} f2c 'const __MFloat8_t' static
//CHECK-NEXT:  | |-FieldDecl {{.*}} f3c 'volatile __MFloat8_t'
//CHECK-NEXT:  | |-AccessSpecDecl {{.*}}
//CHECK-NEXT:  | |-CXXConstructorDecl {{.*}} C1 'void (__MFloat8_t)' implicit-inline
//CHECK-NEXT:  | | |-ParmVarDecl {{.*}} arg '__MFloat8_t'
//CHECK-NEXT:  | | |-CXXCtorInitializer {{.*}} 'f1c' '__MFloat8_t'
//CHECK-NEXT:  | | | `-ImplicitCastExpr {{.*}} '__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  | | |   `-DeclRefExpr {{.*}} '__MFloat8_t' lvalue ParmVar {{.*}} 'arg' '__MFloat8_t'
//CHECK-NEXT:  | | |-CXXCtorInitializer {{.*}} 'f3c' 'volatile __MFloat8_t'
//CHECK-NEXT:  | | | `-ImplicitCastExpr {{.*}} '__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  | | |   `-DeclRefExpr {{.*}} '__MFloat8_t' lvalue ParmVar {{.*}} 'arg' '__MFloat8_t'
//CHECK-NEXT:  | | `-CompoundStmt {{.*}}
//CHECK-NEXT:  | |-CXXMethodDecl {{.*}} func1c '__MFloat8_t (__MFloat8_t)' implicit-inline
//CHECK-NEXT:  | | |-ParmVarDecl {{.*}} arg '__MFloat8_t'
//CHECK-NEXT:  | | `-CompoundStmt {{.*}}
//CHECK-NEXT:  | |   `-ReturnStmt {{.*}}
//CHECK-NEXT:  | |     `-ImplicitCastExpr {{.*}} '__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  | |       `-DeclRefExpr {{.*}} '__MFloat8_t' lvalue ParmVar {{.*}}8 'arg' '__MFloat8_t'
//CHECK-NEXT:  | `-CXXMethodDecl {{.*}} func2c '__MFloat8_t (__MFloat8_t)' static implicit-inline
//CHECK-NEXT:  |   |-ParmVarDecl {{.*}} arg '__MFloat8_t'
//CHECK-NEXT:  |   `-CompoundStmt {{.*}}
//CHECK-NEXT:  |     `-ReturnStmt {{.*}}
//CHECK-NEXT:  |       `-ImplicitCastExpr {{.*}} '__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  |         `-DeclRefExpr {{.*}} '__MFloat8_t' lvalue ParmVar {{.*}} 'arg' '__MFloat8_t'

template <class C> struct S1 {
  C mem1;
};

template <> struct S1<__mfp8> {
  __mfp8 mem2;
};

//CHECK:       |-TemplateArgument type '__MFloat8_t'
//CHECK-NEXT:  | `-BuiltinType {{.*}} '__MFloat8_t'
//CHECK-NEXT:  |-CXXRecordDecl {{.*}} implicit struct S1
//CHECK-NEXT:  `-FieldDecl {{.*}} mem2 '__MFloat8_t'
