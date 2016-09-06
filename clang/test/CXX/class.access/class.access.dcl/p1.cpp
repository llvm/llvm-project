// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// This is just the test for [namespace.udecl]p4 with 'using'
// uniformly stripped out.

// C++03 [namespace.udecl]p4:
//   A using-declaration used as a member-declaration shall refer to a
//   member of a base class of the class being defined, shall refer to
//   a member of an anonymous union that is a member of a base class
//   of the class being defined, or shall refer to an enumerator for
//   an enumeration type that is a member of a base class of the class
//   being defined.

// There is no directly analogous paragraph in C++0x, and the feature
// works sufficiently differently there that it needs a separate test.

namespace test0 {
  namespace NonClass {
    typedef int type;
    struct hiding {};
    int hiding;
    static union { double union_member; };
    enum tagname { enumerator };
  }

  class Test0 {
    NonClass::type; // expected-error {{not a class}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    NonClass::hiding; // expected-error {{not a class}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    NonClass::union_member; // expected-error {{not a class}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    NonClass::enumerator; // expected-error {{not a class}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif
  };
}

struct Opaque0 {};

namespace test1 {
  struct A {
    typedef int type;
    struct hiding {}; // expected-note {{previous use is here}}
    Opaque0 hiding;
    union { double union_member; };
    enum tagname { enumerator };
  };

  struct B : A {
    A::type;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif
    A::hiding;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A::union_member;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A::enumerator;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A::tagname;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    void test0() {
      type t = 0;
    }

    void test1() {
      typedef struct A::hiding local;
      struct hiding _ = local();
    }

    void test2() {
      union hiding _; // expected-error {{tag type that does not match previous}}
    }

    void test3() {
      char array[sizeof(union_member) == sizeof(double) ? 1 : -1];
    }

    void test4() {
      enum tagname _ = enumerator;
    }

    void test5() {
      Opaque0 _ = hiding;
    }
  };
}

namespace test2 {
  struct A {
    typedef int type;
    struct hiding {}; // expected-note {{previous use is here}}
    int hiding;
    union { double union_member; };
    enum tagname { enumerator };
  };

  template <class T> struct B : A {
    A::type;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A::hiding;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A::union_member;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A::enumerator;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A::tagname;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    void test0() {
      type t = 0;
    }

    void test1() {
      typedef struct A::hiding local;
      struct hiding _ = local();
    }

    void test2() {
      union hiding _; // expected-error {{tag type that does not match previous}}
    }

    void test3() {
      char array[sizeof(union_member) == sizeof(double) ? 1 : -1];
    }

    void test4() {
      enum tagname _ = enumerator;
    }

    void test5() {
      Opaque0 _ = hiding;
    }
  };
}

namespace test3 {
  struct hiding {};

  template <class T> struct A {
    typedef int type; // expected-note {{target of using declaration}}
    struct hiding {};
    Opaque0 hiding;
    union { double union_member; };
    enum tagname { enumerator }; // expected-note {{target of using declaration}}
  };

  template <class T> struct B : A<T> {
    A<T>::type; // expected-error {{dependent using declaration resolved to type without 'typename'}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A<T>::hiding;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A<T>::union_member;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A<T>::enumerator;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    A<T>::tagname; // expected-error {{dependent using declaration resolved to type without 'typename'}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    // FIXME: re-enable these when the various bugs involving tags are fixed
#if 0
    void test1() {
      typedef struct A<T>::hiding local;
      struct hiding _ = local();
    }

    void test2() {
      typedef struct A<T>::hiding local;
      union hiding _ = local();
    }
#endif

    void test3() {
      char array[sizeof(union_member) == sizeof(double) ? 1 : -1];
    }

#if 0
    void test4() {
      enum tagname _ = enumerator;
    }
#endif

    void test5() {
      Opaque0 _ = hiding;
    }
  };

  template struct B<int>; // expected-note {{in instantiation}}
}

namespace test4 {
  struct Base {
    int foo();
  };

  struct Unrelated {
    int foo();
  };

  struct Subclass : Base {
  };

  namespace InnerNS {
    int foo();
  }

  // We should be able to diagnose these without instantiation.
  template <class T> struct C : Base {
    InnerNS::foo; // expected-error {{not a class}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    Base::bar; // expected-error {{no member named 'bar'}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    Unrelated::foo; // expected-error {{not a base class}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif

    C::foo; // legal in C++03
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
    // expected-error@-5 {{using declaration refers to its own class}}
#endif

    Subclass::foo; // legal in C++03
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
    // expected-error@-5 {{using declaration refers into 'Subclass::', which is not a base class of 'C'}}
#endif

    int bar();
#if __cplusplus <= 199711L
    //expected-note@-2 {{target of using declaration}}
#endif
    C::bar;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
    // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif
    // expected-error@-6 {{using declaration refers to its own class}}
  };
}

