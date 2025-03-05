// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=ref,both %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -fsyntax-only -verify=expected,both %s


struct Foo {
    constexpr void zomg() const { (void)(1 / 0); } // both-error {{constant expression}} \
                                                      both-warning {{division by zero}} \
                                                      both-note 2{{division by zero}}
};

struct S {
    constexpr S() {}
    constexpr bool operator==(const S&) const { // both-error {{never produces a constant expression}}
      return 1 / 0; // both-warning {{division by zero}} \
                       both-note 3{{division by zero}}
    }

    constexpr bool heh() const {
        auto F = new Foo();
        F->zomg(); // both-note {{in call to 'F->zomg()'}}
        delete F;
        return false;
    }
};

constexpr S s;

static_assert(s.heh()); // both-error {{constant expression}} \
                           both-note {{in call to 's.heh()'}}

constexpr S s2;
constexpr const S *sptr = &s;
constexpr const S *sptr2 = &s2;
static_assert(s == s2); // both-error {{constant expression}} \
                           both-note {{in call to 's.operator==(s2)'}}
static_assert(*sptr == *sptr2); // both-error {{constant expression}} \
                                   both-note {{in call to '*sptr.operator==(s2)'}}

struct A {
  constexpr int foo() { (void)(1/0); return 1;} // both-error {{never produces a constant expression}} \
                                                   both-warning {{division by zero}} \
                                                   both-note 2{{division by zero}}
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
static_assert(d.c->b.a->foo() == 1); // both-error {{constant expression}} \
                                        both-note {{in call to 'd.c->b.a->foo()'}}

template <typename T>
struct Bar {
  template <typename U>
  constexpr int fail1() const { return 1 / 0; } // both-warning {{division by zero}} \
                                                // both-note {{division by zero}}
  template <typename U, int num>
  constexpr int fail2() const { return 1 / 0; } // both-warning {{division by zero}} \
                                                // both-note {{division by zero}}
  template <typename ...Args>
  constexpr int fail3(Args... args) const { return 1 / 0; } // both-warning {{division by zero}} \
                                                // both-note {{division by zero}}
};

constexpr Bar<int> bar;
static_assert(bar.fail1<int>()); // both-error {{constant expression}} \
                                 // both-note {{in call to 'bar.fail1<int>()'}}
static_assert(bar.fail2<int*, 42>()); // both-error {{constant expression}} \
                                      // both-note {{in call to 'bar.fail2<int *, 42>()'}}
static_assert(bar.fail3(3, 4UL, bar, &bar)); // both-error {{constant expression}} \
                                             // expected-note {{in call to 'bar.fail3<int, unsigned long, Bar<int>, const Bar<int> *>(3, 4, &bar, &bar)'}} \
                                             // ref-note {{in call to 'bar.fail3<int, unsigned long, Bar<int>, const Bar<int> *>(3, 4, {}, &bar)'}}



struct MemPtrTest {
  int n;
  void f();
};
MemPtrTest mpt; // both-note {{here}}
constexpr int MemPtr(int (MemPtrTest::*a), void (MemPtrTest::*b)(), int &c) {
  return c; // both-note {{read of non-constexpr variable 'mpt'}}
}
static_assert(MemPtr(&MemPtrTest::n, &MemPtrTest::f, mpt.*&MemPtrTest::n), ""); // both-error {{constant expression}} \
                                                                                // both-note {{in call to 'MemPtr(&MemPtrTest::n, &MemPtrTest::f, mpt.n)'}}
