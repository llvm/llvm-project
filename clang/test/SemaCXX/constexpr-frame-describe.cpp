// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s


struct Foo {
    constexpr void zomg() const { (void)(1 / 0); } // expected-error {{constant expression}} \
                                                      expected-warning {{division by zero}} \
                                                      expected-note 2{{division by zero}}
};

struct S {
    constexpr S() {}
    constexpr bool operator==(const S&) const { // expected-error {{never produces a constant expression}}
      return 1 / 0; // expected-warning {{division by zero}} \
                       expected-note 3{{division by zero}}
    }

    constexpr bool heh() const {
        auto F = new Foo();
        F->zomg(); // expected-note {{in call to 'F->zomg()'}}
        delete F;
        return false;
    }
};

constexpr S s;

static_assert(s.heh()); // expected-error {{constant expression}} \
                           expected-note {{in call to 's.heh()'}}

constexpr S s2;
constexpr const S *sptr = &s;
constexpr const S *sptr2 = &s2;
static_assert(s == s2); // expected-error {{constant expression}} \
                           expected-note {{in call to 's.operator==(s2)'}}
static_assert(*sptr == *sptr2); // expected-error {{constant expression}} \
                                   expected-note {{in call to '*sptr.operator==(s2)'}}

struct A {
  constexpr int foo() { (void)(1/0); return 1;} // expected-error {{never produces a constant expression}} \
                                                   expected-warning {{division by zero}} \
                                                   expected-note 2{{division by zero}}
};

struct B {
  A aa;
  A *a = &aa;
};

struct C {
  B b;
};

struct D {
  C cc;
  C *c = &cc;
};

constexpr D d{};
static_assert(d.c->b.a->foo() == 1); // expected-error {{constant expression}} \
                                        expected-note {{in call to 'd.c->b.a->foo()'}}
