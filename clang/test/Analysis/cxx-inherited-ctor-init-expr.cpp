// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

namespace basic_tests {
struct A {
  int x;
  A(int x): x(x) {}
};

struct B : A {
  using A::A;
};

struct C : B {
  using B::B;
};

void test_B() {
  B b(1);
  clang_analyzer_eval(b.x == 1); // expected-warning{{TRUE}}
}

void test_C() {
  C c(2);
  clang_analyzer_eval(c.x == 2); // expected-warning{{TRUE}}
}
} // namespace basic_tests

namespace arguments_with_constructors {
struct S {
  int x, y;
  S(int x, int y): x(x), y(y) {}
  ~S() {}
};

struct A {
  S s;
  int z;
  A(S s, int z) : s(s), z(z) {}
};

struct B : A {
  using A::A;
};

void test_B() {
  B b(S(1, 2), 3);
  // FIXME: There should be no execution path on which this is false.
  clang_analyzer_eval(b.s.x == 1); // expected-warning{{TRUE}}
                                   // expected-warning@-1{{FALSE}}

  // FIXME: There should be no execution path on which this is false.
  clang_analyzer_eval(b.s.y == 2); // expected-warning{{TRUE}}
                                   // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(b.z == 3); // expected-warning{{TRUE}}
}
} // namespace arguments_with_constructors

namespace inherited_constructor_crash {
class a {
public:
  a(int);
};
struct b : a {
  using a::a; // Ihnerited ctor.
};
void c() {
  int d;
  // This construct expr utilizes the inherited ctor.
  // Note that d must be uninitialized to cause the crash.
  (b(d)); // expected-warning{{1st function call argument is an uninitialized value}}
}
} // namespace inherited_constructor_crash
