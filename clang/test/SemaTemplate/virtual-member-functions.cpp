// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR5557 {
template <class T> struct A {
  A();
  virtual void anchor();
  virtual int a(T x);
};
template<class T> A<T>::A() {}
template<class T> void A<T>::anchor() { }

template<class T> int A<T>::a(T x) { 
  return *x; // expected-error{{requires pointer operand}}
}

void f(A<int> x) {
  x.anchor(); // expected-note{{instantiation}}
}

template<typename T>
struct X {
  virtual void f();
};

template<>
void X<int>::f() { }
}

template<typename T>
struct Base {
  virtual ~Base() { 
    int *ptr = 0;
    T t = ptr; // expected-error{{cannot initialize}}
  }
};

template<typename T>
struct Derived : Base<T> {
  virtual void foo() { }
};

template struct Derived<int>; // expected-note {{in instantiation of member function 'Base<int>::~Base' requested here}}

template<typename T>
struct HasOutOfLineKey {
  HasOutOfLineKey() { } 
  virtual T *f(float *fp);
};

template<typename T>
T *HasOutOfLineKey<T>::f(float *fp) {
  return fp; // expected-error{{cannot initialize return object of type 'int *' with an lvalue of type 'float *'}}
}

HasOutOfLineKey<int> out_of_line; // expected-note{{in instantiation of member function 'HasOutOfLineKey<int>::f' requested here}}

namespace std {
  class type_info;
}

namespace PR7114 {
  class A { virtual ~A(); }; // expected-note{{declared private here}}

  template<typename T>
  class B {
  public:
    class Inner : public A { }; // expected-error{{base class 'PR7114::A' has private destructor}}
    static Inner i;
    static const unsigned value = sizeof(i) == 4;
  };

  int f() { return B<int>::value; }

  void test_typeid(B<float>::Inner bfi) {
    (void)typeid(bfi); // expected-note{{implicit default destructor}}
  }

  template<typename T>
  struct X : A {
    void f() { }
  };

  void test_X(X<int> xi, X<float> xf) {
    xi.f();
  }
}
