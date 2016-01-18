// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X0 {
  static T value;
};

template<typename T>
T X0<T>::value = 0; // expected-error{{no viable conversion}}

struct X1 {
  X1(int);
};

struct X2 { }; // expected-note{{candidate constructor (the implicit copy constructor) not viable}}

int& get_int() { return X0<int>::value; }
X1& get_X1() { return X0<X1>::value; }

double*& get_double_ptr() { return X0<int*>::value; } // expected-error{{non-const lvalue reference to type 'double *' cannot bind to a value of unrelated type 'int *'}}

X2& get_X2() {
  return X0<X2>::value; // expected-note{{instantiation}}
}

template<typename T> T x; // expected-warning{{variable templates are a C++14 extension}}
