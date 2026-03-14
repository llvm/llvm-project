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

template <typename T>
struct Bar {
  template <typename U>
  constexpr int fail1() const { return 1 / 0; } // expected-warning {{division by zero}} \
                                                // expected-note {{division by zero}}
  template <typename U, int num>
  constexpr int fail2() const { return 1 / 0; } // expected-warning {{division by zero}} \
                                                // expected-note {{division by zero}}
  template <typename ...Args>
  constexpr int fail3(Args... args) const { return 1 / 0; } // expected-warning {{division by zero}} \
                                                // expected-note {{division by zero}}
};

constexpr Bar<int> bar;
static_assert(bar.fail1<int>()); // expected-error {{constant expression}} \
                                 // expected-note {{in call to 'bar.fail1<int>()'}}
static_assert(bar.fail2<int*, 42>()); // expected-error {{constant expression}} \
                                      // expected-note {{in call to 'bar.fail2<int *, 42>()'}}
static_assert(bar.fail3(3, 4UL, bar, &bar)); // expected-error {{constant expression}} \
                                             // expected-note {{in call to 'bar.fail3<int, unsigned long, Bar<int>, const Bar<int> *>(3, 4, {}, &bar)'}}
