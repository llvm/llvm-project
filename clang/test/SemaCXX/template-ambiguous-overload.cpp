// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace n {
template <int>
void f(); // expected-note{{explicit instantiation candidate function 'n::f<0>' template here [with $0 = 0]}}

extern template void f<0>();
}

using namespace n;

template <int>
void f() {} // expected-note{{explicit instantiation candidate function 'f<0>' template here [with $0 = 0]}}

template void f<0>(); // expected-error{{partial ordering for explicit instantiation of 'f' is ambiguous}}

