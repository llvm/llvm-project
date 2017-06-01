// RUN: %clang_cc1 -triple i386-pc-win32 -fms-extensions -emit-llvm -fms-volatile -o - < %s | FileCheck %s
struct foo {
  volatile int x;
};
struct bar {
  int x;
};
typedef _Complex float __declspec(align(8)) baz;

#pragma pack(push)
#pragma pack(1)
struct qux {
   volatile int f;
};
#pragma pack(pop)

void test1(struct foo *p, struct foo *q) {
  *p = *q;
  // CHECK-LABEL: @test1
  // CHECK: load atomic volatile {{.*}} acquire
  // CHECK: store atomic volatile {{.*}}, {{.*}} release
}
void test2(volatile int *p, volatile int *q) {
  *p = *q;
  // CHECK-LABEL: @test2
  // CHECK: load atomic volatile {{.*}} acquire
  // CHECK: store atomic volatile {{.*}}, {{.*}} release
}
void test3(struct foo *p, struct foo *q) {
  p->x = q->x;
  // CHECK-LABEL: @test3
  // CHECK: load atomic volatile {{.*}} acquire
  // CHECK: store atomic volatile {{.*}}, {{.*}} release
}
void test4(volatile struct foo *p, volatile struct foo *q) {
  p->x = q->x;
  // CHECK-LABEL: @test4
  // CHECK: load atomic volatile {{.*}} acquire
  // CHECK: store atomic volatile {{.*}}, {{.*}} release
}
void test5(volatile struct foo *p, volatile struct foo *q) {
  *p = *q;
  // CHECK-LABEL: @test5
  // CHECK: load atomic volatile {{.*}} acquire
  // CHECK: store atomic volatile {{.*}}, {{.*}} release
}
void test6(struct bar *p, struct bar *q) {
  *p = *q;
  // CHECK-LABEL: @test6
  // CHECK-NOT: load atomic volatile {{.*}}
  // CHECK-NOT: store atomic volatile {{.*}}, {{.*}}
}
void test7(volatile struct bar *p, volatile struct bar *q) {
  *p = *q;
  // CHECK-LABEL: @test7
  // CHECK: load atomic volatile {{.*}} acquire
  // CHECK: store atomic volatile {{.*}}, {{.*}} release
}
void test8(volatile double *p, volatile double *q) {
  *p = *q;
  // CHECK-LABEL: @test8
  // CHECK: load volatile {{.*}}
  // CHECK: store volatile {{.*}}, {{.*}}
}
void test9(volatile baz *p, baz *q) {
  *p = *q;
  // CHECK-LABEL: @test9
  // CHECK: store volatile {{.*}}, {{.*}}
  // CHECK: store volatile {{.*}}, {{.*}}
}
void test10(volatile long long *p, volatile long long *q) {
  *p = *q;
  // CHECK-LABEL: @test10
  // CHECK: load volatile {{.*}}
  // CHECK: store volatile {{.*}}, {{.*}}
}
void test11(volatile float *p, volatile float *q) {
  *p = *q;
  // CHECK-LABEL: @test11
  // CHECK: load atomic volatile {{.*}} acquire
  // CHECK: store atomic volatile {{.*}}, {{.*}} release
}
int test12(struct qux *p) {
  return p->f;
  // CHECK-LABEL: @test12
  // CHECK: load volatile {{.*}}
}
