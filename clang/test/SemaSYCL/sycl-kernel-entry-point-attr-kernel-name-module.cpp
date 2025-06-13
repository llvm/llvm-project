// Test that SYCL kernel name conflicts that occur across module boundaries are
// properly diagnosed and that declarations are properly merged so that spurious
// conflicts are not reported.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -std=c++17 -fsycl-is-host %t/test.cpp -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -std=c++17 -fsycl-is-device %t/test.cpp -verify

#--- module.modulemap
module M1 { header "m1.h" }
module M2 { header "m2.h" }


#--- common.h
template<int> struct KN;

[[clang::sycl_kernel_entry_point(KN<1>)]]
void common_test1() {}

template<typename T>
[[clang::sycl_kernel_entry_point(T)]]
void common_test2() {}
template void common_test2<KN<2>>();


#--- m1.h
#include "common.h"

[[clang::sycl_kernel_entry_point(KN<3>)]]
void m1_test3() {} // << expected previous declaration note here.

template<typename T>
[[clang::sycl_kernel_entry_point(T)]]
void m1_test4() {} // << expected previous declaration note here.
template void m1_test4<KN<4>>();

[[clang::sycl_kernel_entry_point(KN<5>)]]
void m1_test5() {} // << expected previous declaration note here.

template<typename T>
[[clang::sycl_kernel_entry_point(T)]]
void m1_test6() {} // << expected previous declaration note here.
template void m1_test6<KN<6>>();


#--- m2.h
#include "common.h"

[[clang::sycl_kernel_entry_point(KN<3>)]]
void m2_test3() {} // << expected kernel name conflict here.

template<typename T>
[[clang::sycl_kernel_entry_point(T)]]
void m2_test4() {} // << expected kernel name conflict here.
template void m2_test4<KN<4>>();

[[clang::sycl_kernel_entry_point(KN<7>)]]
void m2_test7() {} // << expected previous declaration note here.

template<typename T>
[[clang::sycl_kernel_entry_point(T)]]
void m2_test8() {} // << expected previous declaration note here.
template void m2_test8<KN<8>>();


#--- test.cpp
#include "m1.h"
#include "m2.h"

// Expected diagnostics for m1_test3() and m2_test3():
// expected-error@m2.h:4 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@m1.h:12 {{previous declaration is here}}

// Expected diagnostics for m1_test4<KN<4>>() and m2_test4<KN<4>>():
// expected-error@m2.h:8 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@m1.h:16 {{previous declaration is here}}

// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@m1.h:4 {{previous declaration is here}}
[[clang::sycl_kernel_entry_point(KN<5>)]]
void test5() {}

// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@m1.h:8 {{previous declaration is here}}
[[clang::sycl_kernel_entry_point(KN<6>)]]
void test6() {}

// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@m2.h:12 {{previous declaration is here}}
[[clang::sycl_kernel_entry_point(KN<7>)]]
void test7() {}

// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@m2.h:16 {{previous declaration is here}}
[[clang::sycl_kernel_entry_point(KN<8>)]]
void test8() {}

void f() {
  common_test1();
  common_test2<KN<2>>();
}
