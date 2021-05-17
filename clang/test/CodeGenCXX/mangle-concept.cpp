// RUN: %clang_cc1 -verify -Wno-return-type -Wno-main -std=c++2a -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s
// expected-no-diagnostics

namespace test1 {
template <bool> struct S {};
template <typename> concept C = true;
template <typename T = int> S<C<T>> f0() { return S<C<T>>{}; }
template S<C<int>> f0<>();
// CHECK: @_ZN5test12f0IiEENS_1SIL_ZNS_1CIT_EEEEEv(
}

template <bool> struct S {};
template <typename> concept C = true;
template <typename T = int> S<C<T>> f0() { return S<C<T>>{}; }
template S<C<int>> f0<>();
// CHECK: @_Z2f0IiE1SIL_Z1CIT_EEEv(
