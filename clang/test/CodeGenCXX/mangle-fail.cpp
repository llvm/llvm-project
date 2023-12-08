// RUN: %clang_cc1 -emit-llvm-only -x c++ -std=c++11 -triple %itanium_abi_triple -verify %s -DN=1
// RUN: %clang_cc1 -emit-llvm-only -x c++ -std=c++11 -triple %itanium_abi_triple -verify %s -DN=2

struct A { int a; };

#if N == 1
// ChooseExpr
template<class T> void test(int (&)[sizeof(__builtin_choose_expr(true, 1, 1), T())]) {} // expected-error {{cannot yet mangle}}
template void test<int>(int (&)[sizeof(int)]);

#elif N == 2
// CompoundLiteralExpr
template<class T> void test(int (&)[sizeof((A){}, T())]) {} // expected-error {{cannot yet mangle}}
template void test<int>(int (&)[sizeof(A)]);

// FIXME: There are several more cases we can't yet mangle.

#else
#error unknown N
#endif
