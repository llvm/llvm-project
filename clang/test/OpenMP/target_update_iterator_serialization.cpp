// Test without serialization:
// RUN: %clang_cc1 -std=c++20 -fopenmp  %s -ast-dump | FileCheck %s

// Test with serialization:
// RUN: %clang_cc1 -std=c++20 -fopenmp  -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++20 -fopenmp -include-pch %t -ast-dump-all /dev/null  \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck %s

// CHECK: OMPTargetUpdateDirective
// CHECK-NEXT: OMPFromClause
// CHECK-NEXT: ArraySubscriptExpr
// CHECK: DeclRefExpr {{.*}} 'a'
// CHECK: DeclRefExpr {{.*}} 'it'


void foo1() {
  int a[10];

#pragma omp target update from(iterator(int it = 0:10) : a[it])
  ;
}

// CHECK: OMPTargetUpdateDirective
// CHECK-NEXT: OMPToClause
// CHECK-NEXT: ArraySubscriptExpr
// CHECK: DeclRefExpr {{.*}} 'a'
// CHECK: DeclRefExpr {{.*}} 'it'

void foo2() {
  int a[10];

#pragma omp target update to(iterator(int it = 0:10) : a[it])
  ;
}
