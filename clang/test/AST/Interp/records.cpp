// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++14 -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple i686 -verify %s
// RUN: %clang_cc1 -verify=ref %s
// RUN: %clang_cc1 -verify=ref -std=c++14 %s
// RUN: %clang_cc1 -verify=ref -triple i686 %s

struct BoolPair {
  bool first;
  bool second;
};

struct Ints {
  int a = 20;
  int b = 30;
  bool c = true;
  BoolPair bp = {true, false};
  int numbers[3] = {1,2,3};

  static const int five = 5;
  static constexpr int getFive() {
    return five;
  }

  constexpr int getTen() const {
    return 10;
  }
};

static_assert(Ints::getFive() == 5, "");

constexpr Ints ints;
static_assert(ints.a == 20, "");
static_assert(ints.b == 30, "");
static_assert(ints.c, "");
static_assert(ints.getTen() == 10, "");
static_assert(ints.numbers[0] == 1, "");
static_assert(ints.numbers[1] == 2, "");
static_assert(ints.numbers[2] == 3, "");

constexpr const BoolPair &BP = ints.bp;
static_assert(BP.first, "");
static_assert(!BP.second, "");
static_assert(ints.bp.first, "");
static_assert(!ints.bp.second, "");


constexpr Ints ints2{-20, -30, false};
static_assert(ints2.a == -20, "");
static_assert(ints2.b == -30, "");
static_assert(!ints2.c, "");

#if __cplusplus >= 201703L
// FIXME: In c++14, this uses a MaterializeTemporaryExpr,
//   which the new interpreter doesn't support yet.
constexpr Ints getInts() {
  return {64, 128, true};
}
constexpr Ints ints3 = getInts();
static_assert(ints3.a == 64, "");
static_assert(ints3.b == 128, "");
static_assert(ints3.c, "");
#endif

constexpr Ints ints4 = {
  .a = 40 * 50,
  .b = 0,
  .c = (ints.a > 0),

};
static_assert(ints4.a == (40 * 50), "");
static_assert(ints4.b == 0, "");
static_assert(ints4.c, "");
static_assert(ints4.numbers[0] == 1, "");
static_assert(ints4.numbers[1] == 2, "");
static_assert(ints4.numbers[2] == 3, "");

constexpr Ints ints5 = ints4;
static_assert(ints5.a == (40 * 50), "");
static_assert(ints5.b == 0, "");
static_assert(ints5.c, "");
static_assert(ints5.numbers[0] == 1, "");
static_assert(ints5.numbers[1] == 2, "");
static_assert(ints5.numbers[2] == 3, "");


struct Ints2 {
  int a = 10;
  int b;
};
// FIXME: Broken in the new constant interpreter.
//   Should be rejected, but without asan errors.
//constexpr Ints2 ints2;

class C {
  public:
    int a;
    int b;

  constexpr C() : a(100), b(200) {}
};

constexpr C c;
static_assert(c.a == 100, "");
static_assert(c.b == 200, "");

constexpr int getB() {
  C c;
  int &j = c.b;

  j = j * 2;

  return c.b;
}
static_assert(getB() == 400, "");

constexpr int getA(const C &c) {
  return c.a;
}
static_assert(getA(c) == 100, "");

constexpr const C* getPointer() {
  return &c;
}
static_assert(getPointer()->a == 100, "");

#if __cplusplus >= 201703L
// FIXME: In c++14, this uses a MaterializeTemporaryExpr,
//   which the new interpreter doesn't support yet.
constexpr C RVOAndParams(const C *c) {
  return C();
}
constexpr C RVOAndParamsResult = RVOAndParams(&c);
#endif

constexpr int locals() {
  C c;
  c.a = 10;

  // Assignment, not an initializer.
  // c = C(); FIXME
  c.a = 10;


  // Assignment, not an initializer.
  //c = RVOAndParams(&c); FIXME

  return c.a;
}
static_assert(locals() == 10, "");

namespace thisPointer {
  struct S {
    constexpr int get12() { return 12; }
  };

  constexpr int foo() { // ref-error {{never produces a constant expression}}
    S *s = nullptr;
    return s->get12(); // ref-note 2{{member call on dereferenced null pointer}} \
                       // expected-note {{member call on dereferenced null pointer}}

  }
  static_assert(foo() == 12, ""); // ref-error {{not an integral constant expression}} \
                                  // ref-note {{in call to 'foo()'}} \
                                  // expected-error {{not an integral constant expression}} \
                                  // expected-note {{in call to 'foo()'}}
};

struct FourBoolPairs {
  BoolPair v[4] = {
    {false, false},
    {false,  true},
    {true,  false},
    {true,  true },
  };
};
// Init
constexpr FourBoolPairs LT;
// Copy ctor
constexpr FourBoolPairs LT2 = LT;
static_assert(LT2.v[0].first == false, "");
static_assert(LT2.v[0].second == false, "");
static_assert(LT2.v[2].first == true, "");
static_assert(LT2.v[2].second == false, "");

class Base {
public:
  int i;
  constexpr Base() : i(10) {}
  constexpr Base(int i) : i(i) {}
};

class A : public Base {
public:
  constexpr A() : Base(100) {}
  constexpr A(int a) : Base(a) {}
};
constexpr A a{};
static_assert(a.i == 100, "");
constexpr A a2{12};
static_assert(a2.i == 12, "");
static_assert(a2.i == 200, ""); // ref-error {{static assertion failed}} \
                                // ref-note {{evaluates to '12 == 200'}} \
                                // expected-error {{static assertion failed}} \
                                // expected-note {{evaluates to '12 == 200'}}


struct S {
  int a = 0;
  constexpr int get5() const { return 5; }
  constexpr void fo() const {
    this; // expected-warning {{expression result unused}} \
          // ref-warning {{expression result unused}}
    this->a; // expected-warning {{expression result unused}} \
             // ref-warning {{expression result unused}}
    get5();
  }

  constexpr int m() const {
    fo();
    return 1;
  }
};
constexpr S s;
static_assert(s.m() == 1, "");

namespace MI {
  class A {
  public:
    int a;
    constexpr A(int a) : a(a) {}
  };

  class B {
  public:
    int b;
    constexpr B(int b) : b(b) {}
  };

  class C : public A, public B {
  public:
    constexpr C() : A(10), B(20) {}
  };
  constexpr C c = {};
  static_assert(c.a == 10, "");
  static_assert(c.b == 20, "");


  class D : private A, private B {
    public:
    constexpr D() : A(20), B(30) {}
    constexpr int getA() const { return a; }
    constexpr int getB() const { return b; }
  };
  constexpr D d = {};
  static_assert(d.getA() == 20, "");
  static_assert(d.getB() == 30, "");
};

namespace DeriveFailures {
  struct Base { // ref-note 2{{declared here}}
    int Val;
  };

  struct Derived : Base {
    int OtherVal;

    constexpr Derived(int i) : OtherVal(i) {} // ref-error {{never produces a constant expression}} \
                                              // ref-note 2{{non-constexpr constructor 'Base' cannot be used in a constant expression}}
  };

  // FIXME: This is currently not being diagnosed with the new constant interpreter.
  constexpr Derived D(12); // ref-error {{must be initialized by a constant expression}} \
                           // ref-note {{in call to 'Derived(12)'}} \
                           // ref-note {{declared here}} \
                           // expected-error {{must be initialized by a constant expression}}
  static_assert(D.Val == 0, ""); // ref-error {{not an integral constant expression}} \
                                 // ref-note {{initializer of 'D' is not a constant expression}}

  struct AnotherBase {
    int Val;
    constexpr AnotherBase(int i) : Val(12 / i) {} //ref-note {{division by zero}} \
                                                  //expected-note {{division by zero}}
  };

  struct AnotherDerived : AnotherBase {
    constexpr AnotherDerived(int i) : AnotherBase(i) {}
  };
  constexpr AnotherBase Derp(0); // ref-error {{must be initialized by a constant expression}} \
                                 // ref-note {{in call to 'AnotherBase(0)'}} \
                                 // expected-error {{must be initialized by a constant expression}} \
                                 // expected-note {{in call to 'AnotherBase(0)'}}

  struct YetAnotherBase {
    int Val;
    constexpr YetAnotherBase(int i) : Val(i) {}
  };

  struct YetAnotherDerived : YetAnotherBase {
    using YetAnotherBase::YetAnotherBase; //ref-note {{declared here}}
    int OtherVal;

    constexpr bool doit() const { return Val == OtherVal; }
  };

  constexpr YetAnotherDerived Oops(0); // ref-error {{must be initialized by a constant expression}} \
                                       // ref-note {{constructor inherited from base class 'YetAnotherBase' cannot be used in a constant expression}} \
                                       // expected-error {{must be initialized by a constant expression}}
                                       // FIXME: Missing reason for rejection.
};
