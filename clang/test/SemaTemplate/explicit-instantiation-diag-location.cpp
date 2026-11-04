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

template <typename T> struct Outer {
  template <typename U> void f(U);
  template <typename U> static U var;
  template <typename U> struct Inner {};
};
template <typename T> template <typename U> void Outer<T>::f(U) {}
template <typename T> template <typename U> U Outer<T>::var = U{};

void use_members() {
  Outer<int> o;
  o.f<double>(1.0);
  (void)Outer<int>::var<double>;
  Outer<int>::Inner<double> inner;
}

template void Outer<int>::f<double>(double);  // expected-note{{previous explicit instantiation is here}}
template void Outer<int>::f<double>(double);  // expected-error{{duplicate explicit instantiation of 'f<double>'}}

template double Outer<int>::var<double>;      // expected-note{{previous explicit instantiation is here}}
template double Outer<int>::var<double>;      // expected-error{{duplicate explicit instantiation of 'var<double>'}}

template struct Outer<int>::Inner<double>;    // expected-note{{previous explicit instantiation is here}}
template struct Outer<int>::Inner<double>;    // expected-error{{duplicate explicit instantiation of 'Inner<double>'}}

template <typename T> struct A {
  template <typename U> struct B {
    template <typename V> void f(V);
  };
};
template <typename T> template <typename U> template <typename V>
void A<T>::B<U>::f(V) {}

void use_nested() { A<int>::B<double> b; b.f<float>(1.0f); }

template void A<int>::B<double>::f<float>(float);  // expected-note{{previous explicit instantiation is here}}
template void A<int>::B<double>::f<float>(float);  // expected-error{{duplicate explicit instantiation of 'f<float>'}}

} // namespace GH21133
