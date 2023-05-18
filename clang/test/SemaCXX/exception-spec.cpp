// RUN: %clang_cc1 -fsyntax-only -verify -fcxx-exceptions -std=c++11 %s

namespace MissingOnTemplate {
  template<typename T> void foo(T) noexcept(true); // expected-note {{previous}}
  template<typename T> void foo(T); // expected-error {{missing exception specification 'noexcept(true)'}}
  void test() { foo(0); }
}

struct UseBeforeComplete1 {
  ~UseBeforeComplete1(); // expected-note {{previous}}
  struct X {
    friend UseBeforeComplete1::~UseBeforeComplete1() noexcept; // expected-warning {{previously declared with an implicit}}
  };
};

struct ThrowingDtor { ~ThrowingDtor() noexcept(false); };
struct UseBeforeComplete2 {
  ~UseBeforeComplete2(); // expected-note {{previous}}
  struct X {
    friend UseBeforeComplete2::~UseBeforeComplete2() noexcept; // expected-error {{does not match previous}}
  };
  ThrowingDtor td;
};

struct UseBeforeComplete3 {
  ~UseBeforeComplete3();
  struct X {
    friend UseBeforeComplete3::~UseBeforeComplete3(); // ok, implicitly noexcept(true)
  };
};
static_assert(noexcept(UseBeforeComplete3()), "");

struct UseBeforeComplete4 {
  ~UseBeforeComplete4();
  struct X {
    friend UseBeforeComplete4::~UseBeforeComplete4(); // ok, implicitly noexcept(false)
  };
  ThrowingDtor td;
};
static_assert(!noexcept(UseBeforeComplete4()), "");

namespace AssignmentOp {
  struct D1;
  struct D2;
  struct B {
    B &operator=(const B&);
    virtual D1 &operator=(const D1&) noexcept; // expected-note {{overridden}}
    virtual D2 &operator=(const D2&) noexcept; // expected-note {{overridden}}
  };
  struct D1 : B {}; // expected-error {{more lax}}
  struct D2 : B {
    D2 &operator=(const D2&); // expected-error {{more lax}}
  };
}
