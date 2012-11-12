// RUN: %clang_cc1 -fsyntax-only -verify %s

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
  t = 17; // expected-error{{incompatible}}
}

void test_intptr(X0<int*> xi, X0<int*>::Inner xii) {
  xi.f(0);
  xii.g(0);
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
