// RUN: %clang_cc1 -verify -std=c++2c -emit-llvm -triple %itanium_abi_triple -o - %s -fclang-abi-compat=latest | FileCheck %s
// expected-no-diagnostics

// FIXME: Is the empty case case correct? These are not defined by the itanium ABI yet.
namespace GH218820 {
// CHECK: define {{.*}}@_ZN8GH2188201fITpTtTyEJEEEvvQfraa1CIiE(
// CHECK: define {{.*}}@_ZN8GH2188201fIJNS_1CEEEEvvQfraa1CIiE(
template <template <typename> concept... C>
void f() requires (C<int> && ...) {}

// CHECK: define {{.*}}@_ZN8GH2188201gITpTtTyEJEEEvvQfraa1VIiE(
// CHECK: define {{.*}}@_ZN8GH2188201gIJNS_1VEEEEvvQfraa1VIiE(
template <template <typename> auto... V>
void g() requires (V<int> && ...) {}

template <typename T>
concept C = true;

template <typename T>
constexpr auto V = true;

void h() {
    f<>();
    f<C>();
    g<>();
    g<V>();
}
}
