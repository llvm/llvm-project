// RUN: %clang_cc1 -std=c++17 -ast-dump %s -ast-dump-filter Test | FileCheck %s

struct A{};

template <typename T>
auto Test(T* pt) {
  // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '*'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  (void)*pt;

  // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '++'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  (void)(++pt);

  // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '+'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  (void)(+pt);

  // CHECK: BinaryOperator {{.*}} '<dependent type>' '+'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
  (void)(pt + 3);

  // CHECK: BinaryOperator {{.*}} '<dependent type>' '-'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  (void)(pt -pt);
}


