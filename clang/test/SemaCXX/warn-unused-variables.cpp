// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -verify %s
template<typename T> void f() {
  T t;
  t = 17;
}

// PR5407
struct A { A(); };
struct B { ~B(); };
void f() {
  A a;
  B b;
}

// PR5531
namespace PR5531 {
  struct A {
  };

  struct B {
    B(int);
  };

  struct C {
    ~C();
  };

  void test() {
    A();
    B(17);
    C();
  }
}

template<typename T>
struct X0 { };

template<typename T>
void test_dependent_init(T *p) {
  X0<int> i(p);
  (void)i;
}

namespace PR6948 {
  template<typename T> class X; // expected-note{{template is declared here}}
  
  void f() {
    X<char> str (read_from_file()); // expected-error{{use of undeclared identifier 'read_from_file'}} \
                                       expected-error{{implicit instantiation of undefined template 'PR6948::X<char>'}}
  }
}

void unused_local_static() {
  static int x = 0;
  static int y = 0; // expected-warning{{unused variable 'y'}}
#pragma unused(x)
}

// PR10168
namespace PR10168 {
  // We expect a warning in the definition only for non-dependent variables, and
  // a warning in the instantiation only for dependent variables.
  template<typename T>
  struct S {
    void f() {
      int a; // expected-warning {{unused variable 'a'}}
      T b; // expected-warning 2{{unused variable 'b'}}
    }
  };

  template<typename T>
  void f() {
    int a; // expected-warning {{unused variable 'a'}}
    T b; // expected-warning 2{{unused variable 'b'}}
  }

  void g() {
    S<int>().f(); // expected-note {{here}}
    S<char>().f(); // expected-note {{here}}
    f<int>(); // expected-note {{here}}
    f<char>(); // expected-note {{here}}
  }
}

namespace PR11550 {
  struct S1 {
    S1();
  };
  S1 makeS1();
  void testS1(S1 a) {
    // This constructor call can be elided.
    S1 x = makeS1(); // expected-warning {{unused variable 'x'}}

    // This one cannot, so no warning.
    S1 y;

    // This call cannot, but the constructor is trivial.
    S1 z = a; // expected-warning {{unused variable 'z'}}
  }

  // The same is true even when we know thet constructor has side effects.
  void foo();
  struct S2 {
    S2() {
      foo();
    }
  };
  S2 makeS2();
  void testS2(S2 a) {
    S2 x = makeS2(); // expected-warning {{unused variable 'x'}}
    S2 y;
    S2 z = a; // expected-warning {{unused variable 'z'}}
  }

  // Or when the constructor is not declared by the user.
  struct S3 {
    S1 m;
  };
  S3 makeS3();
  void testS3(S3 a) {
    S3 x = makeS3(); // expected-warning {{unused variable 'x'}}
    S3 y;
    S3 z = a; // expected-warning {{unused variable 'z'}}
  }
}

namespace ctor_with_cleanups {
  struct S1 {
    ~S1();
  };
  struct S2 {
    S2(const S1&);
  };
  void func() {
    S2 s((S1()));
  }
}
