// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace std {
  typedef decltype(sizeof(int)) size_t;

  template <typename E>
  struct initializer_list
  {
    const E *p;
    size_t n;
    initializer_list(const E *p, size_t n) : p(p), n(n) {}
  };

  struct string {
    string(const char *);
  };

  template<typename A, typename B>
  struct pair {
    pair(const A&, const B&);
  };
}

namespace bullet2 {
  double ad[] = { 1, 2.0 };
  int ai[] = { 1, 2.0 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{override}}

  struct S2 {
    int m1;
    double m2, m3;
  };

  S2 s21 = { 1, 2, 3.0 };
  S2 s22 { 1.0, 2, 3 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{override}}
  S2 s23 { };
}

namespace bullet4_example1 {
  struct S {
    S(std::initializer_list<double> d) {}
    S(std::initializer_list<int> i) {}
    S() {}
  };

  S s1 = { 1.0, 2.0, 3.0 };
  S s2 = { 1, 2, 3 };
  S s3 = { };
}

namespace bullet4_example2 {
  struct Map {
    Map(std::initializer_list<std::pair<std::string,int>>) {}
  };

  Map ship = {{"Sophie",14}, {"Surprise",28}};
}

namespace bullet4_example3 {
  struct S {
    S(int, double, double) {}
    S() {}
  };

  S s1 = { 1, 2, 3.0 };
  // FIXME: This is an ill-formed narrowing initialization.
  S s2 { 1.0, 2, 3 };
  S s3 {};
}

namespace bullet5 {
  struct S {
    S(std::initializer_list<double>) {}
    S(const std::string &) {}
  };

  const S& r1 = { 1, 2, 3.0 };
  const S& r2 = { "Spinach" };
  S& r3 = { 1, 2, 3 };  // expected-error {{non-const lvalue reference to type 'bullet5::S' cannot bind to an initializer list temporary}}
  const int& i1 = { 1 };
  const int& i2 = { 1.1 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{override}} expected-warning {{implicit conversion}}
  const int (&iar)[2] = { 1, 2 };
}

namespace bullet6 {
  int x1 {2};
  int x2 {2.0};  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{override}}
}

namespace bullet7 {
  int** pp {};
}

namespace bullet8 {
  struct A { int i; int j; };
  A a1 { 1, 2 };
  A a2 { 1.2 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{override}} expected-warning {{implicit conversion}}

  struct B {
    B(std::initializer_list<int> i) {}
  };
  B b1 { 1, 2 };
  B b2 { 1, 2.0 };

  struct C {
    C(int i, double j) {}
  };
  C c1 = { 1, 2.2 };
  // FIXME: This is an ill-formed narrowing initialization.
  C c2 = { 1.1, 2 };  // expected-warning {{implicit conversion}}

  int j { 1 };
  int k { };
}
