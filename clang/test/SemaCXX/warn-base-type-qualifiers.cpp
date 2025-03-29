// RUN: %clang_cc1 %s -std=c++11 -Wignored-qualifiers -verify

template <typename T> struct add_const {
  using type = const T;
};
template <typename T> using add_const_t = typename add_const<T>::type;

class A { };

typedef const A A_Const;
class B : public A_Const { }; // expected-warning {{'const' qualifier on base class type 'A_Const' (aka 'const A') has no effect}} \
                              // expected-note {{base class 'A_Const' (aka 'const A') specified here}}

typedef const volatile A A_Const_Volatile;
class C : public A_Const_Volatile { }; // expected-warning {{'const volatile' qualifiers on base class type 'A_Const_Volatile' (aka 'const volatile A') have no effect}} \
                                       // expected-note {{base class 'A_Const_Volatile' (aka 'const volatile A') specified here}}

struct D {
  D(int);
};

template <typename T> struct E : T { // expected-warning {{'const' qualifier on base class type 'const D' has no effect}} \
                                     // expected-note {{base class 'const D' specified here}}
  using T::T;
  E(int &) : E(0) {}
};
E<const D> e(1); // expected-note {{in instantiation of template class 'E<const D>' requested here}}

template <typename T>
struct G : add_const<T>::type { // expected-warning {{'const' qualifier on base class type 'add_const<D>::type' (aka 'const D') has no effect}} \
                                // expected-note {{base class 'add_const<D>::type' (aka 'const D') specified here}}
  using T::T;
  G(int &) : G(0) {}
};
G<D> g(1); // expected-note {{in instantiation of template class 'G<D>' requested here}}
