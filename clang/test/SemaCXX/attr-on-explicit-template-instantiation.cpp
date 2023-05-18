// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

// PR39118
// Make sure that attributes are properly applied to explicit template
// instantiations.

#define HIDDEN __attribute__((__visibility__("hidden")))
#define VISIBLE __attribute__((__visibility__("default")))

namespace ns HIDDEN {
    struct A { };
    template <typename T> struct B { static A a; };
    template <typename T> A B<T>::a;

    // CHECK: @_ZN2ns1BIiE1aE = weak_odr global
    // CHECK-NOT: hidden
    template VISIBLE A B<int>::a;
}

struct C { };
template <typename T> struct D { static C c; };
template <typename T> C D<T>::c;

// CHECK-DAG: @_ZN1DIiE1cB3TAGE
template __attribute__((abi_tag("TAG"))) C D<int>::c;
