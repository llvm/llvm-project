// RUN: %clang_cc1 -fsyntax-only -fopenmp -std=c23 -triple x86_64-unknown-unknown -verify %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-enable-irbuilder -std=c23 -triple x86_64-unknown-unknown -verify %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -std=c23 -triple x86_64-unknown-unknown -verify %s

// RUN: %clang_cc1 -fopenmp -std=c23 -triple x86_64-unknown-unknown -emit-llvm -o - -DCODEGEN %s | FileCheck %s

typedef _BitInt(931) B931;

void sink(B931, B931);

// GH140074
// CHECK-LABEL: define {{.*}}void @gh140074_bound(
// CHECK: call void @__kmpc_for_static_init_{{4|8u?}}(
void gh140074_bound(int a, B931 b) {
#pragma omp for
  for (int i = a; i < b; i++)
    sink(i, b);
}

#ifndef CODEGEN
void gh140074_reduced(int a, B931 b) {
#pragma omp for collapse(2) // expected-note {{as specified in 'collapse' clause}}
  for (int i = a; i < b; i++)
    sink(i, b); // expected-error {{expected 2 for loops after '#pragma omp for', but found only 1}}
}
#endif

// CHECK-LABEL: define {{.*}}void @bitint_iv(
// CHECK: call void @__kmpc_for_static_init_{{4|8u?}}(
void bitint_iv(B931 x) {
  // expected-warning@+2 {{OpenMP loop iteration variable cannot have more than 64 bits size and will be narrowed}}
#pragma omp for
  for (B931 i = 0; i < x; ++i)
    sink(i, x);
}

// CHECK-LABEL: define {{.*}}void @bitint_collapse(
// CHECK: call void @__kmpc_for_static_init_{{4|8u?}}(
void bitint_collapse(B931 a, B931 b, B931 c, B931 d, B931 e, B931 f) {
  // expected-warning@+3 {{OpenMP loop iteration variable cannot have more than 64 bits size and will be narrowed}}
  // expected-warning@+3 {{OpenMP loop iteration variable cannot have more than 64 bits size and will be narrowed}}
#pragma omp for collapse(2)
  for (B931 i = a; i < b; i += c)
    for (B931 j = d; j > e; j += f)
      sink(i, j);
}
