// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s

// PR5908
template <typename Iterator>
void Test(Iterator it) {
  *(it += 1);
}

namespace PR6045 {
  template<unsigned int r>
  class A
  {
    static const unsigned int member = r;
    void f();
  };
  
  template<unsigned int r>
  const unsigned int A<r>::member;
  
  template<unsigned int r>
  void A<r>::f() 
  {
    unsigned k;
    (void)(k % member);
  }
}

namespace PR7198 {
  struct A
  {
    ~A() { }
  };

  template<typename T>
  struct B {
    struct C : A {};
    void f()
    {
      C c = C();
    }
  };
}

namespace PR7724 {
  template<typename OT> int myMethod()
  { return 2 && sizeof(OT); }
}

namespace test4 {
  template <typename T> T *addressof(T &v) {
    return reinterpret_cast<T*>(
             &const_cast<char&>(reinterpret_cast<const volatile char &>(v)));
  }
}

namespace test5 {
  template <typename T> class chained_map {
    int k;
    void lookup() const {
      int &v = (int &)k;
    }
  };
}

namespace test6 {
  template<typename T> T f() {
    const T &v(0);
    return v;
  }
  int use = f<int>();
}

namespace PR8795 {
  template <class _CharT> int test(_CharT t)
  {
    int data [] = {
      sizeof(_CharT) > sizeof(char)
    };
    return data[0];
  }
}

template<typename T> struct CastDependentIntToPointer {
  static void* f() {
    T *x;
    return ((void*)(((unsigned long)(x)|0x1ul)));
  }
};

// Regression test for crasher in r194540.
namespace PR10837 {
  typedef void t(int);
  template<typename> struct A {
    void f();
    static t g;
  };
  t *p;
  template<typename T> void A<T>::f() {
    p = g;
  }
  template struct A<int>;
}

namespace PR18152 {
  template<int N> struct A {
    static const int n = {N};
  };
  template struct A<0>;
}

template<typename T> void stmt_expr_1() {
  static_assert( ({ false; }), "" );
}
void stmt_expr_2() {
  static_assert( ({ false; }), "" ); // expected-error {{failed}}
}

namespace PR45083 {
  struct A { bool x; };

  template<typename> struct B : A {
    void f() {
      const int n = ({ if (x) {} 0; });
    }
  };

  template void B<int>::f();

  template<typename> void f() {
    decltype(({})) x; // expected-error {{incomplete type}}
  }
  template void f<int>(); // expected-note {{instantiation of}}

  template<typename> auto g() {
    auto c = [](auto, int) -> decltype(({})) {};
    using T = decltype(c(0.0, 0));
    using T = void;
    return c(0, 0);
  }
  using U = decltype(g<int>()); // expected-note {{previous}}
  using U = float; // expected-error {{different types ('float' vs 'decltype(g<int>())' (aka 'void'))}}

  void h(auto a, decltype(g<char>())*) {} // expected-note {{previous}}
  void h(auto a, void*) {} // expected-error {{redefinition}}

  void i(auto a) {
    [](auto a, int = ({decltype(a) i; i * 2;})){}(a); // expected-error {{invalid operands to binary expression ('decltype(a)' (aka 'void *') and 'int')}} expected-note {{in instantiation of}}
  }
  void use_i() {
    i(0);
    i((void*)0); // expected-note {{instantiation of}}
  }
}

namespace BindingInStmtExpr {
  template<class ...Ts> struct overload : Ts... {
    overload(Ts ...ts) : Ts(decltype(ts)(ts))... {}
    using Ts::operator()...;
  };

  template<int> struct N {};

  template<class T> auto num_bindings() {
    auto f0 = [](auto t, unsigned) { return N<0>(); };
    auto f1 = [](auto t, int) -> decltype(({ auto [_1] = t; N<1>(); })) { return {}; };
    auto f2 = [](auto t, int) -> decltype(({ auto [_1, _2] = t; N<2>(); })) { return {}; };
    auto f3 = [](auto t, int) -> decltype(({ auto [_1, _2, _3] = t; N<3>(); })) { return {}; };
    return decltype(overload(f0, f1, f2, f3)(T(), 0))();
  }

  struct T { int a; int b; };
  // Make sure we get a correct, non-dependent type back.
  using U = decltype(num_bindings<T>()); // expected-note {{previous}}
  using U = N<3>; // expected-error-re {{type alias redefinition with different types ('N<3>' vs {{.*}}N<2>}}
}
