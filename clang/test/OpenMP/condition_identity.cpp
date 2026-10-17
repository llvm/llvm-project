// RUN: %clang_cc1 -triple x86_64-unknown-linux -fopenmp \
// RUN:   -fopenmp-version=52 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fopenmp \
// RUN:   -fopenmp-version=52 -emit-pch %s -o %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fopenmp \
// RUN:   -fopenmp-version=52 -include-pch %t -emit-llvm %s -o - \
// RUN:   | FileCheck %s

#ifndef HEADER
#define HEADER

int first();
int second();

// Substitution must not equate different parameters with equal values.
#pragma omp declare variant(first) \
    match(user={condition(N)}, implementation={vendor(score(4): llvm)})
#pragma omp declare variant(second) \
    match(user={condition(M)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
template <int N, int M> int distinct();

// The parameter of a class and its out-of-line definition has one binding.
template <int N> struct S {
#pragma omp declare variant(first) \
    match(user={condition(N)}, implementation={vendor(score(4): llvm)})
  static int f();
};
#pragma omp declare variant(second) \
    match(user={condition(N)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
template <int N> int S<N>::f() { return 0; }

// Alias normalization must not add a second identity for the same type.
template <class T, class> using alias = T;
#pragma omp declare variant(first) \
    match(user={condition(sizeof(T) > 1)}, \
          implementation={vendor(score(4): llvm)})
template <class T> int aliases();
#pragma omp declare variant(second) \
    match(user={condition(sizeof(alias<T, T>) > 1)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
template <class T> int aliases();

// A template binding carried only by a destination type also matters.
typedef float F2 __attribute__((ext_vector_type(2)));
typedef int I2 __attribute__((ext_vector_type(2)));
extern F2 input;
template <class T> struct A {
#pragma omp declare variant(first) \
    match(user={condition(sizeof(__builtin_convertvector(input, T)) > 0)}, \
          implementation={vendor(score(4): llvm)})
  friend int types();
};
template <class T> struct B {
#pragma omp declare variant(second) \
    match(user={condition(sizeof(__builtin_convertvector(input, T)) > 0)}, \
          device={kind(cpu)}, implementation={vendor(score(3): llvm)})
  friend int types();
};
template struct A<I2>;
template struct B<I2>;
int types();

// CHECK-LABEL: define {{.*}} @_Z13test_distinctv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_distinct() { return distinct<1, 1>(); }

// CHECK-LABEL: define {{.*}} @_Z19test_redeclarationsv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_redeclarations() { return S<1>::f(); }

// CHECK-LABEL: define {{.*}} @_Z12test_aliasesv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_aliases() { return aliases<int>(); }

// CHECK-LABEL: define {{.*}} @_Z10test_typesv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_types() { return types(); }

#endif
