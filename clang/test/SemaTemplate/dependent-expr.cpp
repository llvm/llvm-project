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
  // GCC doesn't check this: it appears to treat statement-expressions as being
  // value-dependent if they appear in a dependent context, regardless of their
  // contents.
  static_assert( ({ false; }), "" ); // expected-error {{failed}}
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
  template void f<int>();

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
}
