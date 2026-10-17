// RUN: %clang_cc1 -triple x86_64-unknown-linux -fopenmp \
// RUN:   -fopenmp-version=52 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fopenmp \
// RUN:   -fopenmp-version=52 -DBUILD_PCH -emit-pch %s -o %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fopenmp \
// RUN:   -fopenmp-version=52 -DUSE_PCH -include-pch %t -emit-llvm %s -o - \
// RUN:   | FileCheck %s

#ifndef HEADER
#define HEADER

int first();
int second();

// Clang visits the nearest pragma first. Make first the tie-order winner so
// an incorrect subset relationship would select second instead.
#pragma omp declare variant(second) \
    match(user={condition(M)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(N)}, implementation={vendor(score(4): llvm)})
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

#pragma omp declare variant(second) \
    match(user={condition(N)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(N)}, implementation={vendor(score(4): llvm)})
template <int N, int M> int identical_parameters();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(int[M]) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(int[N]) > 0)}, implementation={vendor(score(4): llvm)})
template <int N, int M> int array_parameters();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(int[N]) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(int[N]) > 0)}, implementation={vendor(score(4): llvm)})
template <int N, int M> int identical_array_parameters();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(U (*)()) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(T (*)()) > 0)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int function_results();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(T (*)()) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(T (*)()) > 0)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int identical_function_results();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(void (*)(U)) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(void (*)(T)) > 0)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int function_parameters();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(void (*)(T)) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(void (*)(T)) > 0)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int identical_function_parameters();

template <class T> struct Box {};

#pragma omp declare variant(second) \
    match(user={condition(__is_const(const volatile Box<int>))}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(__is_const(const Box<int>))}, implementation={vendor(score(4): llvm)})
int different_qualifiers();

#pragma omp declare variant(second) \
    match(user={condition(__is_const(const Box<int>))}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(__is_const(const Box<int>))}, implementation={vendor(score(4): llvm)})
int identical_qualifiers();

struct Value { static const int value = 1; };

#pragma omp declare variant(second) \
    match(user={condition(U::value)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(T::value)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int member_parameters();

#pragma omp declare variant(second) \
    match(user={condition(T::value)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(T::value)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int identical_member_parameters();

#pragma omp declare variant(second) \
    match(user={condition(__array_extent(T, 1) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(__array_extent(T, 0) > 0)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int array_dimensions();

#pragma omp declare variant(second) \
    match(user={condition(__array_extent(T, 0) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(__array_extent(T, 0) > 0)}, implementation={vendor(score(4): llvm)})
template <class T, class U> int identical_array_dimensions();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(void (*)(int[M])) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(void (*)(int[N])) > 0)}, implementation={vendor(score(4): llvm)})
template <int N, int M> int parameter_array_bounds();

#pragma omp declare variant(second) \
    match(user={condition(sizeof(void (*)(int[N])) > 0)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(sizeof(void (*)(int[N])) > 0)}, implementation={vendor(score(4): llvm)})
template <int N, int M> int identical_parameter_array_bounds();

#pragma omp declare variant(second) \
    match(user={condition(alias<T, T>::value)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(T::value)}, implementation={vendor(score(4): llvm)})
template <class T> int aliased_member_parameters();

namespace condition_binding { const int value = 1; }
using condition_binding::value;
#pragma omp declare variant(second) \
    match(user={condition(condition_binding::value)}, device={kind(cpu)}, \
          implementation={vendor(score(3): llvm)})
#pragma omp declare variant(first) \
    match(user={condition(value)}, implementation={vendor(score(4): llvm)})
int ordinary_qualification();

#endif

#if !defined(BUILD_PCH) || defined(USE_PCH)

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

// CHECK-LABEL: define {{.*}} @_Z25test_identical_parametersv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_parameters() { return identical_parameters<1, 1>(); }

// CHECK-LABEL: define {{.*}} @_Z21test_array_parametersv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_array_parameters() { return array_parameters<1, 1>(); }

// CHECK-LABEL: define {{.*}} @_Z31test_identical_array_parametersv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_array_parameters() { return identical_array_parameters<1, 1>(); }

// CHECK-LABEL: define {{.*}} @_Z21test_function_resultsv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_function_results() { return function_results<int, int>(); }

// CHECK-LABEL: define {{.*}} @_Z31test_identical_function_resultsv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_function_results() { return identical_function_results<int, int>(); }

// CHECK-LABEL: define {{.*}} @_Z24test_function_parametersv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_function_parameters() { return function_parameters<int, int>(); }

// CHECK-LABEL: define {{.*}} @_Z34test_identical_function_parametersv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_function_parameters() { return identical_function_parameters<int, int>(); }

// CHECK-LABEL: define {{.*}} @_Z25test_different_qualifiersv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_different_qualifiers() { return different_qualifiers(); }

// CHECK-LABEL: define {{.*}} @_Z25test_identical_qualifiersv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_qualifiers() { return identical_qualifiers(); }

// CHECK-LABEL: define {{.*}} @_Z22test_member_parametersv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_member_parameters() { return member_parameters<Value, Value>(); }

// CHECK-LABEL: define {{.*}} @_Z32test_identical_member_parametersv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_member_parameters() { return identical_member_parameters<Value, Value>(); }

// CHECK-LABEL: define {{.*}} @_Z21test_array_dimensionsv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_array_dimensions() { return array_dimensions<int[2][3], int>(); }

// CHECK-LABEL: define {{.*}} @_Z31test_identical_array_dimensionsv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_array_dimensions() { return identical_array_dimensions<int[2][3], int>(); }

// CHECK-LABEL: define {{.*}} @_Z27test_parameter_array_boundsv(
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: call {{.*}} @_Z5firstv
// CHECK-NOT: call {{.*}} @_Z6secondv
// CHECK: ret i32
int test_parameter_array_bounds() { return parameter_array_bounds<1, 1>(); }

// CHECK-LABEL: define {{.*}} @_Z37test_identical_parameter_array_boundsv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_identical_parameter_array_bounds() { return identical_parameter_array_bounds<1, 1>(); }

// CHECK-LABEL: define {{.*}} @_Z30test_aliased_member_parametersv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_aliased_member_parameters() { return aliased_member_parameters<Value>(); }

// CHECK-LABEL: define {{.*}} @_Z27test_ordinary_qualificationv(
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: call {{.*}} @_Z6secondv
// CHECK-NOT: call {{.*}} @_Z5firstv
// CHECK: ret i32
int test_ordinary_qualification() { return ordinary_qualification(); }

#endif
