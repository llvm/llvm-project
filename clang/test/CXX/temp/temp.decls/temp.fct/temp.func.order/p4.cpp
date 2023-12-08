// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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


namespace core_26909 {
  template<typename T> struct A {};
  template<typename T, typename U> void f(T&, U); // expected-note {{candidate}}
  template<typename T, typename U> void f(T&&, A<U>); // expected-note {{candidate}} expected-warning 0-1{{extension}}
  template<typename T, typename U> void g(const T&, U); // expected-note {{candidate}}
  template<typename T, typename U> void g(T&, A<U>); // expected-note {{candidate}}

  void h(int a, const char b, A<int> c) {
    f(a, c); // expected-error{{ambiguous}}
    g(b, c); // expected-error{{ambiguous}}
  }
}

namespace PR22435 {
  template<typename T, typename U> void foo(void (*)(T), const U &); // expected-note {{candidate}}
  template<typename T, typename U> bool foo(void (*)(T &), U &); // expected-note {{candidate}}
  void bar(const int x) { bool b = foo<char>(0, x); } // expected-error {{ambiguous}}
}
