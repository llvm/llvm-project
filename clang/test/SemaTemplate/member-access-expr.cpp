// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T>
void call_f0(T x) {
  x.Base::f0();
}

struct Base {
  void f0();
};

struct X0 : Base { 
  typedef Base CrazyBase;
};

void test_f0(X0 x0) {
  call_f0(x0);
}

template<typename TheBase, typename T>
void call_f0_through_typedef(T x) {
  typedef TheBase Base2;
  x.Base2::f0();
}

void test_f0_through_typedef(X0 x0) {
  call_f0_through_typedef<Base>(x0);
}

template<typename TheBase, typename T>
void call_f0_through_typedef2(T x) {
  typedef TheBase CrazyBase;
#if __cplusplus <= 199711L
  // expected-note@-2 {{lookup from the current scope refers here}}
#endif

  x.CrazyBase::f0(); // expected-error 2{{no member named}}
#if __cplusplus <= 199711L
  // expected-error@-2 {{lookup of 'CrazyBase' in member access expression is ambiguous}}
#endif

}

struct OtherBase { };

struct X1 : Base, OtherBase { 
  typedef OtherBase CrazyBase;
#if __cplusplus <= 199711L
  // expected-note@-2 {{lookup in the object type 'X1' refers here}}
#endif
};

void test_f0_through_typedef2(X0 x0, X1 x1) {
  call_f0_through_typedef2<Base>(x0);
  call_f0_through_typedef2<OtherBase>(x1); // expected-note{{instantiation}}
  call_f0_through_typedef2<Base>(x1); // expected-note{{instantiation}}
}


struct X2 {
  operator int() const;
};

template<typename T, typename U>
T convert(const U& value) {
  return value.operator T(); // expected-error{{operator long}}
}

void test_convert(X2 x2) {
  convert<int>(x2);
  convert<long>(x2); // expected-note{{instantiation}}
}

template<typename T>
void destruct(T* ptr) {
  ptr->~T();
  ptr->T::~T();
}

template<typename T>
void destruct_intptr(int *ip) {
  ip->~T();
  ip->T::~T();
}

void test_destruct(X2 *x2p, int *ip) {
  destruct(x2p);
  destruct(ip);
  destruct_intptr<int>(ip);
}

// PR5220
class X3 {
protected:
  template <int> float* &f0();
  template <int> const float* &f0() const;
  void f1() {
    (void)static_cast<float*>(f0<0>());
  }
  void f1() const{
    (void)f0<0>();
  }
};

// Fun with template instantiation and conversions
struct X4 {
  int& member();
  float& member() const;
};

template<typename T>
struct X5 {
  void f(T* ptr) { int& ir = ptr->member(); }
  void g(T* ptr) { float& fr = ptr->member(); }
};

void test_X5(X5<X4> x5, X5<const X4> x5c, X4 *xp, const X4 *cxp) {
  x5.f(xp);
  x5c.g(cxp);
}

// In theory we can do overload resolution at template-definition time on this.
// We should at least not assert.
namespace test4 {
  struct Base {
    template <class T> void foo() {}
  };

  template <class T> struct Foo : Base {
    void test() {
      foo<int>();
    }
  };
}

namespace test5 {
  template<typename T>
  struct X {
    using T::value;

    T &getValue() {
      return &value;
    }
  };
}

// PR8739
namespace test6 {
  struct A {};
  struct B {};
  template <class T> class Base;
  template <class T> class Derived : public Base<T> {
    A *field;
    void get(B **ptr) {
      // It's okay if at some point we figure out how to diagnose this
      // at instantiation time.
      *ptr = field; // expected-error {{incompatible pointer types assigning to 'test6::B *' from 'test6::A *'}}
    }
  };
}

namespace test7 {
  struct C { void g(); };
  template<typename T> struct A {
    T x;
    static void f() {
      (x.g()); // expected-error {{invalid use of member 'x' in static member function}}
    }
  };
  void h() { A<C>::f(); }
}
