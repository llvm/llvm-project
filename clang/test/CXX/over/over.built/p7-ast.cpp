// RUN: %clang_cc1 -std=c++17 -ast-dump %s -ast-dump-filter Test | FileCheck %s

struct A{};

template <typename T>
auto Test(T* pt) {
  // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '*'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
  return *pt;
}


