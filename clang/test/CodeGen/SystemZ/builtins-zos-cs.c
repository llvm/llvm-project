// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-none-zos %s -emit-llvm -o - | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -triple s390x-linux-gnu %s -verify

int test_cs(unsigned int *a, unsigned int *b, unsigned int c) {
    return __cs(a, b, c);
}
// expected-error@-2 {{call to undeclared function '__cs'; ISO C99 and later do not support implicit function declarations}}

// CHECK-LABEL: define{{.*}} @test_cs
// CHECK: cmpxchg ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
// CHECK: store i32 {{.*}}, ptr {{.*}}, align 4
// CHECK: xor i1 {{.*}}, true

int test_cs1(int *a, int *b, int *c) {
  return __cs1(a, b, c);
}
// expected-error@-2 {{call to undeclared function '__cs1'; ISO C99 and later do not support implicit function declarations}}

// CHECK-LABEL: define{{.*}} @test_cs1
// CHECK: cmpxchg ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
// CHECK: store i32 {{.*}}, ptr {{.*}}, align 4
// CHECK: xor i1 {{.*}}, true

typedef struct __attribute__((__aligned__(8))) {
  int a;
  int b;
} double_word;

int test_cds1(double_word *a, double_word *b, double_word *c) {
  return __cds1(a, b, c);
}
// expected-error@-2 {{call to undeclared function '__cds1'; ISO C99 and later do not support implicit function declarations}}

// CHECK-LABEL: define{{.*}} @test_cds1
// CHECK: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8
// CHECK: store i64 {{.*}}, ptr {{.*}}, align 8
// CHECK: xor i1 {{.*}}, true

typedef struct __attribute__((__aligned__(16))) {
  int a;
  int b;
  int c;
  int d;
} quadruple_word;

int test_cdsg(quadruple_word *a, quadruple_word *b, quadruple_word *c) {
  return __cdsg(a, b, c);
}
// expected-error@-2 {{call to undeclared function '__cdsg'; ISO C99 and later do not support implicit function declarations}}

// CHECK-LABEL: define{{.*}} @test_cdsg
// CHECK: cmpxchg ptr {{.*}}, i128 {{.*}}, i128 {{.*}} seq_cst seq_cst, align 16
// CHECK: store i128 {{.*}}, ptr {{.*}}, align 16
// CHECK: xor i1 {{.*}}, true


int test_csg(double_word *a, double_word *b, double_word *c) {
  return __csg(a, b, c);
}
// expected-error@-2 {{call to undeclared function '__csg'; ISO C99 and later do not support implicit function declarations}}

// CHECK-LABEL: define{{.*}} @test_csg
// CHECK: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8
// CHECK: store i64 {{.*}}, ptr {{.*}}, align 8
// CHECK: xor i1 {{.*}}, true
