// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class A> int x(A x) { return x++; }
int y() { return x<int>(1); }

namespace PR5880 {
  template<typename T>
  struct A { 
    static const int a  = __builtin_offsetof(T, a.array[5].m); // expected-error{{no member named 'a' in 'HasM'}}
  };
  struct HasM {
    float m;
  };

  struct ArrayOfHasM {
    HasM array[10];
  };

  struct B { ArrayOfHasM a; };
  A<B> x;
  A<HasM> x2; // expected-note{{in instantiation of}}

  template<typename T>
  struct AnonymousUnion {
    union {
      int i;
      float f;
    };
  };

  template<typename T>
  void test_anon_union() {
    int array1[__builtin_offsetof(AnonymousUnion<T>, f) == 0? 1 : -1];
    int array2[__builtin_offsetof(AnonymousUnion<int>, f) == 0? 1 : -1];
  }

  template void test_anon_union<int>();
}

namespace AddrOfClassMember {
  template <typename T> struct S {
    int n;
    static void f() {
      +T::n; // expected-error {{invalid use of member}}
    }
  };
  void g() { S<S<int> >::f(); } // expected-note {{in instantiation of}}
}
