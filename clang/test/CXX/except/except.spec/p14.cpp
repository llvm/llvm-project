// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -verify -std=c++11 %s
struct A { };
struct B { };
struct C { };

// Destructor
struct X0 { 
  virtual ~X0() throw(A); // expected-note{{overridden virtual function is here}} 
};
struct X1 { 
  virtual ~X1() throw(B); // expected-note{{overridden virtual function is here}} 
};
struct X2 : public X0, public X1 { }; // expected-error 2{{exception specification of overriding function is more lax than base version}}
 
// Copy-assignment operator.
struct CA0 {
  CA0 &operator=(const CA0&) throw(A);
};
struct CA1 {
  CA1 &operator=(const CA1&) throw(B);
};
struct CA2 : CA0, CA1 { };

void test_CA() {
  CA2 &(CA2::*captr1)(const CA2&) throw(A, B) = &CA2::operator=;
  CA2 &(CA2::*captr2)(const CA2&) throw(A, B, C) = &CA2::operator=;
  CA2 &(CA2::*captr3)(const CA2&) throw(A) = &CA2::operator=; // expected-error{{target exception specification is not superset of source}}
  CA2 &(CA2::*captr4)(const CA2&) throw(B) = &CA2::operator=; // expected-error{{target exception specification is not superset of source}}
}

// In-class member initializers.
struct IC0 {
  int inClassInit = 0;
};
struct IC1 {
  int inClassInit = (throw B(), 0);
};
// FIXME: the exception specification on the default constructor is wrong:
// we cannot currently compute the set of thrown types.
static_assert(noexcept(IC0()), "IC0() does not throw");
static_assert(!noexcept(IC1()), "IC1() throws");

namespace PR13381 {
  struct NoThrowMove {
    NoThrowMove(const NoThrowMove &);
    NoThrowMove(NoThrowMove &&) noexcept;
    NoThrowMove &operator=(const NoThrowMove &) const;
    NoThrowMove &operator=(NoThrowMove &&) const noexcept;
  };
  struct NoThrowMoveOnly {
    NoThrowMoveOnly(NoThrowMoveOnly &&) noexcept;
    NoThrowMoveOnly &operator=(NoThrowMoveOnly &&) noexcept;
  };
  struct X {
    const NoThrowMove a;
    NoThrowMoveOnly b;

    static X val();
    static X &ref();
  };
  // These both perform a move, but that copy might throw, because it calls
  // NoThrowMove's copy constructor (because PR13381::a is const).
  static_assert(!noexcept(X(X::val())), "");
  static_assert(!noexcept(X::ref() = X::val()), "");
}

namespace PR14141 {
  // Part of DR1351: the implicit exception-specification is noexcept(false) if
  // the set of potential exceptions of the special member function contains
  // "any". Hence it is compatible with noexcept(false).
  struct ThrowingBase {
    ThrowingBase() noexcept(false);
    ThrowingBase(const ThrowingBase&) noexcept(false);
    ThrowingBase(ThrowingBase&&) noexcept(false);
    ThrowingBase &operator=(const ThrowingBase&) noexcept(false);
    ThrowingBase &operator=(ThrowingBase&&) noexcept(false);
    ~ThrowingBase() noexcept(false);
  };
  struct Derived : ThrowingBase {
    Derived() noexcept(false) = default;
    Derived(const Derived&) noexcept(false) = default;
    Derived(Derived&&) noexcept(false) = default;
    Derived &operator=(const Derived&) noexcept(false) = default;
    Derived &operator=(Derived&&) noexcept(false) = default;
    ~Derived() noexcept(false) = default;
  };
  struct Derived2 : ThrowingBase {
    Derived2() = default;
    Derived2(const Derived2&) = default;
    Derived2(Derived2&&) = default;
    Derived2 &operator=(const Derived2&) = default;
    Derived2 &operator=(Derived2&&) = default;
    ~Derived2() = default;
  };
  struct Derived3 : ThrowingBase {
    Derived3() noexcept(true) = default; // expected-error {{does not match the calculated}}
    Derived3(const Derived3&) noexcept(true) = default; // expected-error {{does not match the calculated}}
    Derived3(Derived3&&) noexcept(true) = default; // expected-error {{does not match the calculated}}
    Derived3 &operator=(const Derived3&) noexcept(true) = default; // expected-error {{does not match the calculated}}
    Derived3 &operator=(Derived3&&) noexcept(true) = default; // expected-error {{does not match the calculated}}
    ~Derived3() noexcept(true) = default; // expected-error {{does not match the calculated}}
  };
}

namespace rdar13017229 {
  struct Base {
    virtual ~Base() {}
  };
  
  struct Derived : Base {
    virtual ~Derived();
    Typo foo(); // expected-error{{unknown type name 'Typo'}}
  };
}

namespace InhCtor {
  template<int> struct X {};
  struct Base {
    Base(X<0>) noexcept(true);
    Base(X<1>) noexcept(false);
    Base(X<2>) throw(X<2>);
    template<typename T> Base(T) throw(T);
  };
  template<typename T> struct Throw {
    Throw() throw(T);
  };
  struct Derived : Base, Throw<X<3>> {
    using Base::Base;
    Throw<X<4>> x;
  };
  struct Test {
    friend Derived::Derived(X<0>) throw(X<3>, X<4>);
    friend Derived::Derived(X<1>) noexcept(false);
    friend Derived::Derived(X<2>) throw(X<2>, X<3>, X<4>);
  };
  static_assert(!noexcept(Derived{X<5>{}}), "");
}
