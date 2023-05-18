// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X0 {
  void f();
  
  template<typename U>
  void g(U);
  
  struct Nested {
  };
  
  static T member;
};

int &use_X0_int(X0<int> x0i,  // expected-note{{implicit instantiation first required here}}
                int i) {
  x0i.f(); // expected-note{{implicit instantiation first required here}}
  x0i.g(i); // expected-note{{implicit instantiation first required here}}
  X0<int>::Nested nested; // expected-note{{implicit instantiation first required here}}
  return X0<int>::member; // expected-note{{implicit instantiation first required here}}
}

template<>
void X0<int>::f() { // expected-error{{after instantiation}}
}

template<> template<>
void X0<int>::g(int) { // expected-error{{after instantiation}}
}

template<>
struct X0<int>::Nested { }; // expected-error{{after instantiation}}

template<>
int X0<int>::member = 17; // expected-error{{after instantiation}}

template<>
struct X0<int> { }; // expected-error{{after instantiation}}

// Example from the standard
template<class T> class Array { /* ... */ }; 

template<class T> void sort(Array<T>& v) { /* ... */ }

struct String {};

void f(Array<String>& v) {
  
  sort(v); // expected-note{{required}}
           // use primary template 
           // sort(Array<T>&), T is String
}

template<> void sort<String>(Array<String>& v); // // expected-error{{after instantiation}}
template<> void sort<>(Array<char*>& v);	// OK: sort<char*> not yet used

namespace PR6160 {
  template<typename T> void f(T);
  template<> void f(int);
  extern template void f(int);
  template<> void f(int) { }
}
