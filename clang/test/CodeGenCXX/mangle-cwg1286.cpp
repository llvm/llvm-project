// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NEW
// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -emit-llvm -fclang-abi-compat=23 -o - %s | FileCheck %s --check-prefixes=CHECK,OLD

template <template <class> class> struct X {};
template <class> struct Y {};
template <class T> using Z = Y<T>;
template <class T> using W = const Y<T>;

// NEW: define {{.*}}void @_Z1f1XI1YE(
// OLD: define {{.*}}void @_Z1f1XI1ZE(
void f(X<Z>) {}

// CHECK: define {{.*}}void @_Z1g1XI1WE(
void g(X<W>) {}
