// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/module.modulemap %t/use.cpp -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -DDEFINE_LOCALLY -std=c++20 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/module.modulemap %t/use.cpp -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s

//--- module.modulemap
module a { header "a.h" export * }
module b { header "b.h" export * }
module c { header "c.h" export * }

//--- nonmodular.h
void not_constant();

template<typename T> struct A {
  template<T M> static inline T N = [] { not_constant(); return M; } ();
};

template<typename T, T M> inline T N = [] { not_constant(); return M; } ();

template<typename T, T M> inline auto L = [] {};

template<typename T> int Z;

// These lambdas should not be merged, despite having the same context decl and
// mangling number (but different signatures).
inline auto MultipleLambdas = ((void)[](int*) { return 1; }, [] { return 2; });

//--- a.h
#include "nonmodular.h"

//--- b.h
#include "a.h"

int b1() { return A<int>::N<1>; }
int b2() { return N<int, 1>; }

inline auto x1 = L<int, 1>;
inline auto x2 = L<int, 2>;

inline constexpr int *P = &Z<decltype([] { static int n; return &n; }())>;
inline constexpr int *xP = P;

static_assert(!__is_same(decltype(x1), decltype(x2)));

//--- c.h
#include "a.h"

int c1() { return A<int>::N<2>; }
int c2() { return N<int, 2>; }

inline auto y2 = L<int, 2>;
inline auto y1 = L<int, 1>;

inline constexpr int *P = &Z<decltype([] { static int n; return &n; }())>;
inline constexpr int *yP = P;

//--- use.cpp
#ifdef DEFINE_LOCALLY
#include "nonmodular.h"

inline constexpr int *P = &Z<decltype([] { static int n; return &n; }())>;
inline constexpr int *zP = P;

auto z0 = L<int, 0>;
auto z2 = L<int, 2>;
auto z1 = L<int, 1>;
#endif

#include "b.h"
#include "c.h"

int b1v = b1();
int b2v = b2();
int c1v = c1();
int c2v = c2();

// We should merge together matching lambdas.
static_assert(__is_same(decltype(x1), decltype(y1)));
static_assert(__is_same(decltype(x2), decltype(y2)));
static_assert(!__is_same(decltype(x1), decltype(x2)));
static_assert(!__is_same(decltype(y1), decltype(y2)));
static_assert(!__is_same(decltype(x1), decltype(y2)));
static_assert(!__is_same(decltype(x2), decltype(y1)));
static_assert(xP == yP);
#ifdef DEFINE_LOCALLY
static_assert(!__is_same(decltype(x1), decltype(z0)));
static_assert(!__is_same(decltype(x2), decltype(z0)));
static_assert(__is_same(decltype(x1), decltype(z1)));
static_assert(__is_same(decltype(x2), decltype(z2)));
static_assert(xP == zP);
#endif

static_assert(MultipleLambdas() == 2);

// We should not merge the instantiated lambdas from `b.h` and `c.h` together,
// even though they will both have anonymous declaration number #1 within
// A<int> and within the TU, respectively.

// CHECK-LABEL: define {{.*}}global_var_init{{.*}} comdat($_Z1NIiLi1EE) {
// CHECK: load i8, ptr @_ZGV1NIiLi1EE, align 8
// CHECK: call {{.*}} i32 @_ZNK1NIiLi1EEMUlvE_clEv(
// CHECK: store i32 {{.*}}, ptr @_Z1NIiLi1EE

// CHECK-LABEL: define {{.*}}global_var_init{{.*}} comdat($_ZN1AIiE1NILi1EEE) {
// CHECK: load i8, ptr @_ZGVN1AIiE1NILi1EEE, align 8
// CHECK: call {{.*}} i32 @_ZNK1AIiE1NILi1EEMUlvE_clEv(
// CHECK: store i32 {{.*}}, ptr @_ZN1AIiE1NILi1EEE

// CHECK-LABEL: define {{.*}}global_var_init{{.*}} comdat($_Z1NIiLi2EE) {
// CHECK: load i8, ptr @_ZGV1NIiLi2EE, align 8
// CHECK: call {{.*}} i32 @_ZNK1NIiLi2EEMUlvE_clEv(
// CHECK: store i32 {{.*}}, ptr @_Z1NIiLi2EE

// CHECK-LABEL: define {{.*}}global_var_init{{.*}} comdat($_ZN1AIiE1NILi2EEE) {
// CHECK: load i8, ptr @_ZGVN1AIiE1NILi2EEE, align 8
// CHECK: call {{.*}} i32 @_ZNK1AIiE1NILi2EEMUlvE_clEv(
// CHECK: store i32 {{.*}}, ptr @_ZN1AIiE1NILi2EEE

