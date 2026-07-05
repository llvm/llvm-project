// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
#endif

int& get_int() { return X0<int>::value; }
X1& get_X1() { return X0<X1>::value; }

double*& get_double_ptr() { return X0<int*>::value; } // expected-error{{non-const lvalue reference to type 'double *' cannot bind to a value of unrelated type 'int *'}}

X2& get_X2() {
  return X0<X2>::value; // expected-note{{instantiation}}
}

template<typename T> T x; // expected-warning 0-1{{variable templates are a C++14 extension}}
