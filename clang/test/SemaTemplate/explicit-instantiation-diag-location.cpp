// RUN: %clang_cc1 -fsyntax-only -verify %s

// Verify that the "previous explicit instantiation" note points to the first
// explicit instantiation statement, not to the implicit instantiation site.

namespace GH21133 {

template <typename T> void f() {}
void use_f() { f<int>(); }
template void f<int>();          // expected-note{{previous explicit instantiation is here}}
template void f<int>();          // expected-error{{duplicate explicit instantiation of 'f<int>'}}

template <typename T> struct S {};
void use_S(S<int>) {}
template struct S<int>;          // expected-note{{previous explicit instantiation is here}}
template struct S<int>;          // expected-error{{duplicate explicit instantiation of 'S<int>'}}

template <typename T> T var = T{};
int use_var = var<int>;
template int var<int>;           // expected-note{{previous explicit instantiation is here}}
template int var<int>;           // expected-error{{duplicate explicit instantiation of 'var<int>'}}

namespace ns {
  template <typename T> void g() {}
}
void use_g() { ns::g<double>(); }
template void ns::g<double>();   // expected-note{{previous explicit instantiation is here}}
template void ns::g<double>();   // expected-error{{duplicate explicit instantiation of 'g<double>'}}

} // namespace GH21133
