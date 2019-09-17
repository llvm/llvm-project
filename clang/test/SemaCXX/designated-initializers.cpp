// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wno-reorder -Wno-c99-designator -Winitializer-overrides %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wno-reorder -Wno-c99-designator -Woverride-init %s

template <typename T> struct Foo {
  struct SubFoo {
    int bar1;
    int bar2;
  };

  static void Test() { SubFoo sf = {.bar1 = 10, .bar2 = 20}; } // Expected no warning
};

void foo() {
  Foo<int>::Test();
  Foo<bool>::Test();
  Foo<float>::Test();
}

template <typename T> struct Bar {
  struct SubFoo {
    int bar1;
    int bar2;
  };

  static void Test() { SubFoo sf = {.bar1 = 10,    // expected-note 2 {{previous initialization is here}}
                                    .bar1 = 20}; } // expected-warning 2 {{initializer overrides prior initialization of this subobject}}
};

void bar() {
  Bar<int>::Test();  // expected-note {{in instantiation of member function 'Bar<int>::Test' requested here}}
  Bar<bool>::Test(); // expected-note {{in instantiation of member function 'Bar<bool>::Test' requested here}}
}

namespace Reorder {
  struct X {
    X(int n);
  private:
    int i;
  };

  struct foo {
    X x;
    X y;
  };

  foo n = {.y = 4, .x = 5};
  X arr[2] = {[1] = 1, [0] = 2};
}

namespace Reorder2 {
  struct S {
    S();
    S(const S &);
    ~S();
  };

  struct EF {
    S s;
  };

  struct PN {
    PN(const PN &);
  };
  extern PN pn;

  struct FLN {
    EF ef;
    int it;
    PN pn;
  };

  void f() {
    FLN new_elem = {
        .ef = EF(),
        .pn = pn,
        .it = 0,
    };
  }
}

namespace Reorder3 {
  struct S {
    int a, &b, &c; // expected-note 2{{here}}
  };
  S s1 = {
    .a = 1, .c = s1.a, .b = s1.a
  };
  S s2 = {
    .a = 1, .c = s2.a
  }; // expected-error {{uninitialized}}
  S s3 = {
    .b = s3.a, .a = 1,
  }; // expected-error {{uninitialized}}
}

// Check that we don't even think about whether holes in a designated
// initializer are zero-initializable if the holes are filled later.
namespace NoCheckingFilledHoles {
  template<typename T> struct Error { using type = typename T::type; }; // expected-error 3{{'::'}}

  template<int N>
  struct DefaultInitIsError {
    DefaultInitIsError(Error<int[N]> = {}); // expected-note 3{{instantiation}} expected-note 3{{passing}}
    DefaultInitIsError(int, int);
  };

  template<int N>
  struct X {
    int a;
    DefaultInitIsError<N> e;
    int b;
  };
  X<1> x1 = {
    .b = 2,
    .a = 1,
    {4, 4}
  };
  X<2> x2 = {
    .e = {4, 4},
    .b = 2,
    .a = 1
  };
  X<3> x3 = {
    .b = 2,
    .a = 1
  }; // expected-note {{default function argument}}
  X<4> x4 = {
    .a = 1,
    .b = 2
  }; // expected-note {{default function argument}}
  X<5> x5 = {
    .e = {4, 4},
    .a = 1,
    .b = 2
  };
  X<6> x6 = {
    .a = 1,
    .b = 2,
    .e = {4, 4}
  };

  template<int N> struct Y { X<N> x; };
  Y<7> y7 = {
    .x = {.a = 1, .b = 2}, // expected-note {{default function argument}}
    .x.e = {3, 4}
  };
  Y<8> y8 = {
    .x = {.e = {3, 4}},
    .x.a = 1,
    .x.b = 2
  };
}

namespace LargeArrayDesignator {
  struct X {
    int arr[1000000000];
  };
  struct Y {
    int arr[3];
  };
  void f(X x);
  void f(Y y) = delete;
  void g() {
    f({.arr[4] = 1});
  }
}

namespace ADL {
  struct A {};
  void f(A, int);

  namespace X {
    void f(A, int);
    // OK. Would fail if checking {} against type A set the type of the
    // initializer list to A, because ADL would find ADL::f, resulting in
    // ambiguity.
    void g() { f({}, {}); }
  }
}
