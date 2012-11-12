// RUN: %clang_cc1 -fsyntax-only -verify %s

struct C { };

template<typename T>
struct X0 {
  T value; // expected-error{{incomplete}}
};

// Explicitly instantiate a class template specialization
template struct X0<int>;
template struct X0<void>; // expected-note{{instantiation}}

// Explicitly instantiate a function template specialization
template<typename T>
void f0(T t) {
  ++t; // expected-error{{cannot increment}}
}

template void f0(int);
template void f0<long>(long);
template void f0<>(unsigned);
template void f0(int C::*); // expected-note{{instantiation}}

// Explicitly instantiate a member template specialization
template<typename T>
struct X1 {
  template<typename U>
  struct Inner {
    T member1;
    U member2; // expected-error{{incomplete}}
  };
  
  template<typename U>
  void f(T& t, U u) {
    t = u; // expected-error{{incompatible}}
  }
};

template struct X1<int>::Inner<float>;
template struct X1<int>::Inner<double>;
template struct X1<int>::Inner<void>; // expected-note{{instantiation}}

template void X1<int>::f(int&, float);
template void X1<int>::f<long>(int&, long);
template void X1<int>::f<>(int&, double);
template void X1<int>::f<>(int&, int*); // expected-note{{instantiation}}

// Explicitly instantiate members of a class template
struct Incomplete; // expected-note{{forward declaration}}
struct NonDefaultConstructible { // expected-note{{candidate constructor (the implicit copy constructor) not viable}}
  NonDefaultConstructible(int); // expected-note{{candidate constructor}}
};

template<typename T, typename U>
struct X2 {
  void f(T &t, U u) { 
    t = u; // expected-error{{incompatible}}
  }
  
  struct Inner {
    T member1;
    U member2; // expected-error{{incomplete}}
  };
  
  static T static_member1;
  static U static_member2;
};

template<typename T, typename U>
T X2<T, U>::static_member1 = 17; // expected-error{{cannot initialize}}

template<typename T, typename U>
U X2<T, U>::static_member2; // expected-error{{no matching}}

template void X2<int, float>::f(int &, float);
template void X2<int, float>::f(int &, double); // expected-error{{does not refer}}
template void X2<int, int*>::f(int&, int*); // expected-note{{instantiation}}

template struct X2<int, float>::Inner;
template struct X2<int, Incomplete>::Inner; // expected-note{{instantiation}}

template int X2<int, float>::static_member1;
template int* X2<int*, float>::static_member1; // expected-note{{instantiation}}
template 
  NonDefaultConstructible X2<NonDefaultConstructible, int>::static_member1;

template 
  NonDefaultConstructible X2<int, NonDefaultConstructible>::static_member2; // expected-note{{instantiation}}
