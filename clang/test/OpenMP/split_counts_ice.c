/* `counts` operands as ICEs: macros, enumerators, sizeof (not only raw literals). */
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

#define M1 2

extern void body(int);

// CHECK-LABEL: define {{.*}} @from_macros
// CHECK: .split.iv.0
// CHECK: icmp slt i32 {{.*}}, 2
// CHECK: .split.iv.1
// CHECK: icmp slt i32 {{.*}}, 10
void from_macros(void) {
#pragma omp split counts(M1, omp_fill)
  for (int i = 0; i < 10; ++i)
    body(i);
}

enum { EFirst = 3 };

// CHECK-LABEL: define {{.*}} @from_enum
// CHECK: .split.iv.0
// CHECK: icmp slt i32 {{.*}}, 3
// CHECK: .split.iv.1
// CHECK: icmp slt i32 {{.*}}, 10
void from_enum(void) {
#pragma omp split counts(EFirst, omp_fill)
  for (int i = 0; i < 10; ++i)
    body(i);
}

// CHECK-LABEL: define {{.*}} @from_sizeof
// CHECK: .split.iv.0
// CHECK: icmp slt i32 {{.*}}, 1
// CHECK: .split.iv.1
// CHECK: icmp slt i32 {{.*}}, 10
void from_sizeof(void) {
#pragma omp split counts(sizeof(char), omp_fill)
  for (int i = 0; i < 10; ++i)
    body(i);
}

// CHECK-LABEL: define {{.*}} @from_macro_expr
// CHECK: .split.iv.0
// CHECK: icmp slt i32 {{.*}}, 4
// CHECK: .split.iv.1
// CHECK: icmp slt i32 {{.*}}, 10
#define BASE 1
void from_macro_expr(void) {
#pragma omp split counts(BASE + 3, omp_fill)
  for (int i = 0; i < 10; ++i)
    body(i);
}
