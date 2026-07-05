// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

extern int bounds1(int);
extern int bounds2(int);

extern void fun2(int n, int *a, int *b);
extern void fun3(int n, int *a, int *b);

void fun1(int n, int *a, int *b)
{
#pragma omp task depend(iterator(j = 0 : bounds1(n)), in : a[b[j]])
  {
    fun2(n, a, b);
  }
// CHECK: alloca %struct.kmp_depend_info, i64 [[FIRST_VLA:%.*]], align 16

#pragma omp task depend(iterator(j = 0 : bounds2(n)), in : a[b[j]])
  {
    fun3(n, a, b);
  }
// CHECK-NOT: alloca %struct.kmp_depend_info, i64 [[FIRST_VLA]], align 16
}
