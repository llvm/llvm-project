// RUN: %clang_cc1 -std=c++17 -ast-dump %s -ast-dump-filter Test | FileCheck %s

struct A{};

template <typename T, typename U>
auto Test(T* pt, U* pu) {
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
  (void)(pt - pt);

  // CHECK: BinaryOperator {{.*}} '<dependent type>' '-'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'U *' lvalue ParmVar {{.*}} 'pu' 'U *'
  (void)(pt - pu);

  // CHECK: BinaryOperator {{.*}} '<dependent type>' '=='
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'U *' lvalue ParmVar {{.*}} 'pu' 'U *'
  (void)(pt == pu);

}


