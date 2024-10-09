// RUN: %clang_cc1 -emit-llvm-only -x c++ -std=c++11 -triple %itanium_abi_triple -verify %s -DN=1
// RUN: %clang_cc1 -emit-llvm-only -x c++ -std=c++11 -triple %itanium_abi_triple -verify %s -DN=2
// RUN: %clang_cc1 -emit-llvm-only -x c++ -std=c++11 -triple arm64-apple-ios -fptrauth-intrinsics -verify %s -DN=3
// RUN: %clang_cc1 -emit-llvm-only -x c++ -std=c++11 -triple aarch64-linux-gnu -fptrauth-intrinsics -verify %s -DN=3

struct A { int a; };

#if N == 1
// ChooseExpr
template<class T> void test(int (&)[sizeof(__builtin_choose_expr(true, 1, 1), T())]) {} // expected-error {{cannot yet mangle}}
template void test<int>(int (&)[sizeof(int)]);

#elif N == 2
// CompoundLiteralExpr
template<class T> void test(int (&)[sizeof((A){}, T())]) {} // expected-error {{cannot yet mangle}}
template void test<int>(int (&)[sizeof(A)]);

#elif N == 3
// __builtin_ptrauth_type_discriminator
template <class T, unsigned disc>
struct S1 {};

template<class T>
void func(S1<T, __builtin_ptrauth_type_discriminator(T)> s1) { // expected-error {{cannot yet mangle __builtin_ptrauth_type_discriminator expression}}
}

void testfunc1() {
  func(S1<int(), __builtin_ptrauth_type_discriminator(int())>());
}

// FIXME: There are several more cases we can't yet mangle.

#else
#error unknown N
#endif
