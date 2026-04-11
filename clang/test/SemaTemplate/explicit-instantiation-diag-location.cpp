// RUN: %clang_cc1 -fsyntax-only -verify %s

// Verify that the "previous explicit instantiation" note points to the first
// explicit instantiation statement, not to the implicit instantiation site.
// This is a regression test for https://github.com/llvm/llvm-project/issues/21133

// Function template with implicit instantiation before explicit.
template <typename T> void f() {}
void use_f() { f<int>(); }
template void f<int>();          // expected-note{{previous explicit instantiation is here}}
template void f<int>();          // expected-error{{duplicate explicit instantiation of 'f<int>'}}

// Class template with implicit instantiation before explicit.
template <typename T> struct S {};
void use_S(S<int>) {}
template struct S<int>;          // expected-note{{previous explicit instantiation is here}}
template struct S<int>;          // expected-error{{duplicate explicit instantiation of 'S<int>'}}

// Cross-namespace: template in ns, explicit instantiation at global scope.
namespace ns {
  template <typename T> void g() {}
}
void use_g() { ns::g<double>(); }
template void ns::g<double>();   // expected-note{{previous explicit instantiation is here}}
template void ns::g<double>();   // expected-error{{duplicate explicit instantiation of 'g<double>'}}

// Variable template with implicit instantiation before explicit.
template <typename T> T var = T{};
int use_var = var<int>;
template int var<int>;           // expected-note{{previous explicit instantiation is here}}
template int var<int>;           // expected-error{{duplicate explicit instantiation of 'var<int>'}}
