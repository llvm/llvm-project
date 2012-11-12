// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class T> struct A { A(); };
template<class T> int &f(T); 
template<class T> float &f(T*); 
template<class T> double &f(const T*);

template<class T> void g(T); // expected-note{{candidate}}
template<class T> void g(T&); // expected-note{{candidate}}

template<class T> int &h(const T&); 
template<class T> float &h(A<T>&);

void m() { 
  const int *p; 
  double &dr1 = f(p); 
  float x; 
  g(x); // expected-error{{ambiguous}}
  A<int> z; 
  float &fr1 = h(z);
  const A<int> z2; 
  int &ir1 = h(z2);
}
