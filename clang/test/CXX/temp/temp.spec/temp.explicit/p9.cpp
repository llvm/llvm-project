// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T>
struct X0 {
  void f(T &t) {
    t = 1; // expected-error{{incompatible integer to pointer conversion}}
  }
  
  void g(T &t);
  
  void h(T &t);
  
  static T static_var;
};

template<typename T>
inline void X0<T>::g(T & t) {
  t = 1; // expected-error{{incompatible integer to pointer conversion}}
}

template<typename T>
void X0<T>::h(T & t) {
  t = 1;
}

template<typename T>
T X0<T>::static_var = 1;

extern template struct X0<int*>;

int *&test(X0<int*> xi, int *ip) {
  xi.f(ip); // expected-note{{instantiation}}
  xi.g(ip); // expected-note{{instantiation}}
  xi.h(ip);
  return X0<int*>::static_var;
}

template<typename T>
void f0(T& t) {
  t = 1; // expected-error{{incompatible integer to pointer conversion}}
}

template<typename T>
inline void f1(T& t) {
  t = 1; // expected-error 2{{incompatible integer to pointer conversion}}
}

extern template void f0<>(int *&);
extern template void f1<>(int *&);

void test_f0(int *ip, float *fp) {
  f0(ip);
  f0(fp); // expected-note{{instantiation}}
}

void test_f1(int *ip, float *fp) {
  f1(ip); // expected-note{{instantiation}}
  f1(fp); // expected-note{{instantiation}}
}
