// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
class C { C(int a0 = 0); };

template<>
C<char>::C(int a0);

struct S { }; // expected-note 3 {{candidate constructor (the implicit copy constructor)}}

template<typename T> void f1(T a, T b = 10) { } // expected-error{{no viable conversion}} \
// expected-note{{passing argument to parameter 'b' here}}

template<typename T> void f2(T a, T b = T()) { }

template<typename T> void f3(T a, T b = T() + T()); // expected-error{{invalid operands to binary expression ('S' and 'S')}}

void g() {
  f1(10);
  f1(S()); // expected-note{{in instantiation of default function argument expression for 'f1<S>' required here}}
  
  f2(10);
  f2(S());
  
  f3(10);
  f3(S()); // expected-note{{in instantiation of default function argument expression for 'f3<S>' required here}}
}

template<typename T> struct F {
  F(T t = 10); // expected-error{{no viable conversion}} \
  // expected-note{{passing argument to parameter 't' here}}
  void f(T t = 10); // expected-error{{no viable conversion}} \
  // expected-note{{passing argument to parameter 't' here}}
};

struct FD : F<int> { };

void g2() {
  F<int> f;
  FD fd;
}

void g3(F<int> f, F<struct S> s) {
  f.f();
  s.f(); // expected-note{{in instantiation of default function argument expression for 'f<S>' required here}}
  
  F<int> f2;
  F<S> s2; // expected-note{{in instantiation of default function argument expression for 'F<S>' required here}}
}

template<typename T> struct G {
  G(T) {}
};

void s(G<int> flags = 10) { }

// Test default arguments
template<typename T>
struct X0 {
  void f(T = T()); // expected-error{{no matching}}
};

template<typename U>
void X0<U>::f(U) { }

void test_x0(X0<int> xi) {
  xi.f();
  xi.f(17);
}

struct NotDefaultConstructible { // expected-note 2{{candidate}}
  NotDefaultConstructible(int); // expected-note 2{{candidate}}
};

void test_x0_not_default_constructible(X0<NotDefaultConstructible> xn) {
  xn.f(NotDefaultConstructible(17));
  xn.f(42);
  xn.f(); // expected-note{{in instantiation of default function argument}}
}

template<typename T>
struct X1 {
  typedef T value_type;
  X1(const value_type& value = value_type());
};

void test_X1() {
  X1<int> x1;
}

template<typename T>
struct X2 {
  void operator()(T = T()); // expected-error{{no matching}}
};

void test_x2(X2<int> x2i, X2<NotDefaultConstructible> x2n) {
  x2i();
  x2i(17);
  x2n(NotDefaultConstructible(17));
  x2n(); // expected-note{{in instantiation of default function argument}}
}

// PR5283
namespace PR5283 {
template<typename T> struct A {
  A(T = 1); // expected-error 3 {{cannot initialize a parameter of type 'int *' with an rvalue of type 'int'}} \
  // expected-note 3{{passing argument to parameter here}}
};

struct B : A<int*> { 
  B();
};
B::B() { } // expected-note {{in instantiation of default function argument expression for 'A<int *>' required he}}

struct C : virtual A<int*> {
  C();
};
C::C() { } // expected-note {{in instantiation of default function argument expression for 'A<int *>' required he}}

struct D {
  D();
  
  A<int*> a;
};
D::D() { } // expected-note {{in instantiation of default function argument expression for 'A<int *>' required he}}
}

// PR5301
namespace pr5301 {
  void f(int, int = 0);

  template <typename T>
  void g(T, T = 0);

  template <int I>
  void i(int a = I);

  template <typename T>
  void h(T t) {
    f(0);
    g(1);
    g(t);
    i<2>();
  }

  void test() {
    h(0);
  }
}

// PR5810
namespace PR5810 {
  template<typename T>
  struct allocator {
    allocator() { int a[sizeof(T) ? -1 : -1]; } // expected-error2 {{array with a negative size}}
  };
  
  template<typename T>
  struct vector {
    vector(const allocator<T>& = allocator<T>()) {} // expected-note2 {{instantiation of}}
  };
  
  struct A { };
  struct B { };

  template<typename>
  void FilterVTs() {
    vector<A> Result;
  }
  
  void f() {
    vector<A> Result;
  }

  template<typename T>
  struct X {
    vector<B> bs;
    X() { }
  };

  void f2() {
    X<float> x; // expected-note{{member function}}
  }
}

template<typename T> void f4(T, int = 17);
template<> void f4<int>(int, int);

void f4_test(int i) {
  f4(i);
}

// Instantiate for initialization
namespace InstForInit {
  template<typename T>
  struct Ptr {
    typedef T* type;
    Ptr(type);
  };

  template<typename T>
  struct Holder {
    Holder(int i, Ptr<T> ptr = 0);
  };

  void test_holder(int i) {
    Holder<int> h(i);
  }
};

namespace PR5810b {
  template<typename T>
  T broken() {
    T t;
    double**** not_it = t;
  }

  void f(int = broken<int>());
  void g() { f(17); }
}

namespace PR5810c {
  template<typename T>
  struct X { 
    X() { 
      T t;
      double *****p = t; // expected-error{{cannot initialize a variable of type 'double *****' with an lvalue of type 'int'}}
    }
    X(const X&) { }
  };

  struct Y : X<int> { // expected-note{{instantiation of}}
  };

  void f(Y y = Y());

  void g() { f(); }
}

namespace PR8127 {
  template< typename T > class PointerClass {
  public:
    PointerClass( T * object_p ) : p_( object_p ) {
      p_->acquire();
    }
  private:    
    T * p_;
  };

  class ExternallyImplementedClass;

  class MyClass {
    void foo( PointerClass<ExternallyImplementedClass> = 0 );
  };
}

namespace rdar8427926 {
  template<typename T>
  struct Boom {
    ~Boom() {
      T t;
      double *******ptr = t; // expected-error 2{{cannot initialize}}
    }
  };

  Boom<float> *bfp;

  struct X {
    void f(Boom<int> = Boom<int>()) { } // expected-note{{requested here}}
    void g(int x = (delete bfp, 0)); // expected-note{{requested here}}
  };

  void test(X *x) {
    x->f();
    x->g();
  }
}

namespace PR8401 {
  template<typename T> 
  struct A { 
    A() { T* x = 1; } // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
  };

  template<typename T>
  struct B {
    B(const A<T>& a = A<T>()); // expected-note{{in instantiation of}}
  };

  void f(B<int> b = B<int>());

  void g() {
    f();
  }
}

namespace PR12581 {
  const int a = 0;
  template < typename > struct A;
  template < typename MatrixType, int =
  A < MatrixType >::Flags ? : A < MatrixType >::Flags & a > class B;
  void
  fn1 ()
  {
  }
}

namespace PR13758 {
  template <typename T> struct move_from {
    T invalid;
  };
  template <class K>
  struct unordered_map {
    explicit unordered_map(int n = 42);
    unordered_map(move_from<K> other);
  };
  template<typename T>
  void StripedHashTable() {
    new unordered_map<void>();
    new unordered_map<void>;
  }
  void tt() {
    StripedHashTable<int>();
  }
}
