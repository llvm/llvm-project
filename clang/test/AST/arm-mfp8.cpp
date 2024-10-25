// RUN: %clang_cc1 -std=c++11 -triple aarch64-arm-none-eabi -target-feature -fp8 -ast-dump %s | \
// RUN:  FileCheck %s --strict-whitespace

// REQUIRES: aarch64-registered-target || arm-registered-target

/*  Various contexts where type __mfp8 can appear. */

#include<arm_neon.h>
/*  Namespace */
namespace {
  __mfp8 f2n;
  __mfp8 arr1n[10];
}

//CHECK:       |-NamespaceDecl {{.*}}
//CHECK-NEXT:  | |-VarDecl {{.*}} f2n '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | `-VarDecl {{.*}} arr1n '__mfp8[10]'


  const __mfp8 func1n(const __mfp8 mfp8) {
    // this should fail
    __mfp8 f1n;
    f1n  = mfp8;
    return f1n;
  }
//CHECK:    |-FunctionDecl {{.*}} func1n 'const __mfp8 (const __mfp8)'
//CHECK:            | `-VarDecl {{.*}} f1n '__mfp8':'__MFloat8_t'
//CHECK-NEXT:       |-BinaryOperator {{.*}} '__mfp8':'__MFloat8_t' lvalue '='
//CHECK-NEXT:       | |-DeclRefExpr {{.*}} '__mfp8':'__MFloat8_t' lvalue Var {{.*}} 'f1n' '__mfp8':'__MFloat8_t'
//CHECK-NEXT:       | `-ImplicitCastExpr {{.*}} '__mfp8':'__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:       |   `-DeclRefExpr {{.*}} 'const __mfp8':'const __MFloat8_t' lvalue ParmVar {{.*}} 'mfp8' 'const __mfp8':'const __MFloat8_t'
//CHECK-NEXT:        `-ReturnStmt {{.*}}
//CHECK-NEXT:         `-ImplicitCastExpr {{.*}} '__mfp8':'__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:           `-DeclRefExpr {{.*}} '__mfp8':'__MFloat8_t' lvalue Var {{.*}} 'f1n' '__mfp8':'__MFloat8_t'


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
//CHECK-NEXT:  | |-FieldDecl {{.*}} f1c '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | |-VarDecl {{.*}} f2c 'const __mfp8':'const __MFloat8_t' static
//CHECK-NEXT:  | |-FieldDecl {{.*}} f3c 'volatile __MFloat8_t'
//CHECK-NEXT:  | |-AccessSpecDecl {{.*}}
//CHECK-NEXT:  | |-CXXConstructorDecl {{.*}} C1 'void (__mfp8)' implicit-inline
//CHECK-NEXT:  | | |-ParmVarDecl {{.*}} arg '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | | |-CXXCtorInitializer {{.*}} 'f1c' '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | | | `-ImplicitCastExpr {{.*}} '__mfp8':'__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  | | |   `-DeclRefExpr {{.*}} '__mfp8':'__MFloat8_t' lvalue ParmVar {{.*}} 'arg' '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | | |-CXXCtorInitializer {{.*}} 'f3c' 'volatile __MFloat8_t'
//CHECK-NEXT:  | | | `-ImplicitCastExpr {{.*}} '__mfp8':'__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  | | |   `-DeclRefExpr {{.*}} '__mfp8':'__MFloat8_t' lvalue ParmVar {{.*}} 'arg' '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | | `-CompoundStmt {{.*}}
//CHECK-NEXT:  | |-CXXMethodDecl {{.*}} func1c '__mfp8 (__mfp8)' implicit-inline
//CHECK-NEXT:  | | |-ParmVarDecl {{.*}} arg '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | | `-CompoundStmt {{.*}}
//CHECK-NEXT:  | |   `-ReturnStmt {{.*}}
//CHECK-NEXT:  | |     `-ImplicitCastExpr {{.*}} '__mfp8':'__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  | |       `-DeclRefExpr {{.*}} '__mfp8':'__MFloat8_t' lvalue ParmVar {{.*}}8 'arg' '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  | `-CXXMethodDecl {{.*}} func2c '__mfp8 (__mfp8)' static implicit-inline
//CHECK-NEXT:  |   |-ParmVarDecl {{.*}} arg '__mfp8':'__MFloat8_t'
//CHECK-NEXT:  |   `-CompoundStmt {{.*}}
//CHECK-NEXT:  |     `-ReturnStmt {{.*}}
//CHECK-NEXT:  |       `-ImplicitCastExpr {{.*}} '__mfp8':'__MFloat8_t' <LValueToRValue>
//CHECK-NEXT:  |         `-DeclRefExpr {{.*}} '__mfp8':'__MFloat8_t' lvalue ParmVar {{.*}} 'arg' '__mfp8':'__MFloat8_t'

template <class C> struct S1 {
  C mem1;
};

template <> struct S1<__mfp8> {
  __mfp8 mem2;
};

//CHECK:       |-TemplateArgument type '__MFloat8_t'
//CHECK-NEXT:  | `-BuiltinType {{.*}} '__MFloat8_t'
//CHECK-NEXT:  |-CXXRecordDecl {{.*}} implicit struct S1
//CHECK-NEXT:  `-FieldDecl {{.*}} mem2 '__mfp8':'__MFloat8_t'
