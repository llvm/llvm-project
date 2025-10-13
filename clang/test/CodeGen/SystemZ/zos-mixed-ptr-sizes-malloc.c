// RUN: %clang_cc1 -triple s390x-ibm-zos -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=X64
#include <stddef.h>
void *__malloc31(size_t);

int test_1() {
  // X64-LABEL: define {{.*}} i32 @test_1()
  // X64: ret i32 %add20
  int *__ptr32 a;
  int *b;
  int i;
  int sum1, sum2, sum3;

  a = (int *__ptr32)__malloc31(sizeof(int) * 10);

  b = a;
  sum1 = 0;
  for (i = 0; i < 10; ++i) {
    a[i] = i;
    sum1 += i;
  }

  sum2 = 0;
  for (i = 0; i < 10; ++i) {
    sum2 += a[i];
  }
  sum3 = 0;
  for (i = 0; i < 10; ++i) {
    sum3 += b[i];
  }

  return (sum1 + sum2 + sum3);
}

int test_2() {
  // X64-LABEL: define {{.*}} i32 @test_2()
  // X64: ret i32 4
  int *a = (int *)__malloc31(sizeof(int));
  int *__ptr32 b;

  *a = 99;
  b = a;
  *b = 44;

  // Test should return 4
  return (*b - 40);
}

int test_3() {
  // X64-LABEL: define {{.*}} i32 @test_3()
  // X64: ret i32 4
  int *a = (int *)__malloc31(sizeof(int));
  int *__ptr32 b;

  *a = 99;
  b = a;

  // Test should return 4
  return (*b - 95);
}

int test_4() {
  // X64-LABEL: define {{.*}} i32 @test_4()
  // X64: ret i32 1
  int *a = (int *)__malloc31(sizeof(int));
  float *d = (float *)__malloc31(sizeof(float));

  int *__ptr32 b;
  int *c;

  float *__ptr32 e;
  float *f;

  *a = 0;
  *d = 0.0;

  b = a;
  c = a;
  e = d;
  f = d;

  // Test should return 1
  return (b == c && e == f);
}

