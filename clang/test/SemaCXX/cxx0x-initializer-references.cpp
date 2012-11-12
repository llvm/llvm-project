// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct one { char c; };
struct two { char c[2]; };

namespace reference {
  struct A {
    int i1, i2;
  };

  void single_init() {
    const int &cri1a = {1};
    const int &cri1b{1};

    int i = 1;
    int &ri1a = {i};
    int &ri1b{i};

    int &ri2 = {1}; // expected-error {{cannot bind to an initializer list temporary}}

    A a{1, 2};
    A &ra1a = {a};
    A &ra1b{a};
  }

  void reference_to_aggregate() {
    const A &ra1{1, 2};
    A &ra2{1, 2}; // expected-error {{cannot bind to an initializer list temporary}}

    const int (&arrayRef)[] = {1, 2, 3};
    static_assert(sizeof(arrayRef) == 3 * sizeof(int), "bad array size");
  }

  struct B {
    int i1;
  };

  void call() {
    void f(const int&);
    f({1});

    void g(int&); // expected-note {{passing argument}}
    g({1}); // expected-error {{cannot bind to an initializer list temporary}}
    int i = 0;
    g({i});

    void h(const B&);
    h({1});

    void a(B&); // expected-note {{passing argument}}
    a({1}); // expected-error {{cannot bind to an initializer list temporary}}
    B b{1};
    a({b});
  }

  void overloading() {
    one f(const int&);
    two f(const B&);

    // First is identity conversion, second is user-defined conversion.
    static_assert(sizeof(f({1})) == sizeof(one), "bad overload resolution");

    one g(int&);
    two g(const B&);

    static_assert(sizeof(g({1})) == sizeof(two), "bad overload resolution");

    one h(const int&);
    two h(const A&);

    static_assert(sizeof(h({1, 2})) == sizeof(two), "bad overload resolution");
  }

  void edge_cases() {
    // FIXME: very poor error message
    int const &b({0}); // expected-error {{could not bind}}
  }

}

namespace PR12182 {
  void f(int const(&)[3]);

  void g() {
      f({1, 2});
  }
}

namespace PR12660 {
  const int &i { 1 };
  struct S { S(int); } const &s { 2 };
}
