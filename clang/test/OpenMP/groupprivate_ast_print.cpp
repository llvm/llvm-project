// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -triple x86_64-apple-darwin10.6.0 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-linux-gnu -ast-dump %s | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -triple x86_64-apple-darwin10.6.0 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -triple x86_64-unknown-linux-gnu -ast-dump %s | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct St{
 int a;
};

struct St1{
 int a;
 static int b;
#pragma omp groupprivate(b)
};
// CHECK: VarDecl {{.*}} b 'int' static
// CHECK-NEXT: OMPGroupPrivateDeclAttr
// CHECK-NEXT: OMPGroupPrivateDecl
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'b' 'int'

int a, b;
#pragma omp groupprivate(a)
#pragma omp groupprivate(b)
// CHECK: VarDecl {{.*}} a 'int'
// CHECK-NEXT: OMPGroupPrivateDeclAttr
// CHECK: VarDecl {{.*}} b 'int'
// CHECK-NEXT: OMPGroupPrivateDeclAttr
// CHECK-NEXT: OMPGroupPrivateDecl
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int'
// CHECK-NEXT: OMPGroupPrivateDecl
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'b' 'int'

template <class T> T foo() {
  static T v;
  #pragma omp groupprivate(v)
  return v;
}
// CHECK: OMPGroupPrivateDecl
// CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue Var {{.*}} 'v' 'T'

int main () {
  static int a;
#pragma omp groupprivate(a)
  a=2;
  return (foo<int>());
}
// CHECK: VarDecl {{.*}} a 'int' static
// CHECK-NEXT: OMPGroupPrivateDeclAttr
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OMPGroupPrivateDecl
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int'

#endif
