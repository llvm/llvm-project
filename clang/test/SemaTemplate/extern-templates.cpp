// RUN: %clang_cc1 -triple i686-pc-win32 -fsyntax-only -verify %s -DMS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu-pc-win32 -fsyntax-only -verify %s

template<typename T>
class X0 {
public:
  void f(T t);
  
  struct Inner {
    void g(T t);
  };
};

template<typename T>
void X0<T>::f(T t) {
  t = 17; // expected-error{{incompatible}}
}

extern template class X0<int>;

extern template class X0<int*>;

template<typename T>
void X0<T>::Inner::g(T t) {
#ifdef MS
  t = 17; // expected-error{{assigning to 'long *' from incompatible}} expected-error{{assigning to 'int *' from incompatible}}
#else
  t = 17; // expected-error{{assigning to 'long *' from incompatible}}
#endif
}

void test_intptr(X0<int*> xi, X0<int*>::Inner xii) {
  xi.f(0);
#ifdef MS
  xii.g(0); // expected-note {{instantiation}}
#else
  xii.g(0);
#endif
}

extern template class X0<long*>; 

void test_longptr(X0<long*> xl, X0<long*>::Inner xli) {
  xl.f(0);
  xli.g(0);
}

template class X0<long*>; // expected-note 2{{instantiation}}

template<typename T>
class X1 {
public:
  void f(T t) { t += 2; }
  
  void g(T t);
};

template<typename T>
void X1<T>::g(T t) { 
  t += 2; 
}

extern template class X1<void*>;

void g_X1(X1<void*> x1, void *ptr) {
  x1.g(ptr);
}

extern template void X1<const void*>::g(const void*);

void g_X1_2(X1<const void *> x1, const void *ptr) {
  x1.g(ptr);
}

namespace static_const_member {
  template <typename T> struct A { static const int n; };
  template <typename T> const int A<T>::n = 3;
  extern template struct A<int>;
  int arr[A<int>::n];
}
