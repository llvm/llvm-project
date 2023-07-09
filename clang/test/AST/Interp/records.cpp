// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++14 -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple i686 -verify %s
// RUN: %clang_cc1 -verify=ref %s
// RUN: %clang_cc1 -verify=ref -std=c++14 %s
// RUN: %clang_cc1 -verify=ref -std=c++20 %s
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

constexpr Ints getInts() {
  return {64, 128, true};
}
constexpr Ints ints3 = getInts();
static_assert(ints3.a == 64, "");
static_assert(ints3.b == 128, "");
static_assert(ints3.c, "");

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
constexpr Ints2 ints22; // expected-error {{without a user-provided default constructor}} \
                        // expected-error {{must be initialized by a constant expression}} \
                        // ref-error {{without a user-provided default constructor}}

constexpr Ints2 I2 = Ints2{12, 25};
static_assert(I2.a == 12, "");
static_assert(I2.b == 25, "");

class C {
  public:
    int a;
    int b;

  constexpr C() : a(100), b(200) {}

  constexpr C get() const {
    return *this;
  }
};

constexpr C c;
static_assert(c.a == 100, "");
static_assert(c.b == 200, "");

constexpr C c2 = C().get();
static_assert(c2.a == 100, "");
static_assert(c2.b == 200, "");

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

constexpr C RVOAndParams(const C *c) {
  return C();
}
constexpr C RVOAndParamsResult = RVOAndParams(&c);

/// Parameter and return value have different types.
constexpr C RVOAndParams(int a) {
  return C();
}
constexpr C RVOAndParamsResult2 = RVOAndParams(12);

class Bar { // expected-note {{definition of 'Bar' is not complete}} \
            // ref-note {{definition of 'Bar' is not complete}}
public:
  constexpr Bar(){}
  constexpr Bar b; // expected-error {{cannot be constexpr}} \
                   // expected-error {{has incomplete type 'const Bar'}} \
                   // ref-error {{cannot be constexpr}} \
                   // ref-error {{has incomplete type 'const Bar'}}
};
constexpr Bar B; // expected-error {{must be initialized by a constant expression}} \
                 // expected-error {{failed to evaluate an expression}} \
                 // ref-error {{must be initialized by a constant expression}}
constexpr Bar *pb = nullptr;

constexpr int locals() {
  C c;
  c.a = 10;

  // Assignment, not an initializer.
  c = C();
  c.a = 10;


  // Assignment, not an initializer.
  c = RVOAndParams(&c);

  return c.a;
}
static_assert(locals() == 100, "");

namespace thisPointer {
  struct S {
    constexpr int get12() { return 12; }
  };

  constexpr int foo() { // ref-error {{never produces a constant expression}} \
                        // expected-error {{never produces a constant expression}}
    S *s = nullptr;
    return s->get12(); // ref-note 2{{member call on dereferenced null pointer}} \
                       // expected-note 2{{member call on dereferenced null pointer}}

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
    getInts();
  }

  constexpr int m() const {
    fo();
    return 1;
  }
};
constexpr S s;
static_assert(s.m() == 1, "");

namespace InitializerTemporaries {
  class Bar {
  private:
    int a;

  public:
    constexpr Bar() : a(10) {}
    constexpr int getA() const { return a; }
  };

  class Foo {
  public:
    int a;

    constexpr Foo() : a(Bar().getA()) {}
  };
  constexpr Foo F;
  static_assert(F.a == 10, "");


  /// Needs constexpr destructors.
#if __cplusplus >= 202002L
  /// Does
  ///    Arr[Pos] = Value;
  ///    ++Pos;
  /// in its destructor.
  class BitSetter {
  private:
    int *Arr;
    int &Pos;
    int Value;

  public:
    constexpr BitSetter(int *Arr, int &Pos, int Value) :
      Arr(Arr), Pos(Pos), Value(Value) {}

    constexpr int getValue() const { return 0; }
    constexpr ~BitSetter() {
      Arr[Pos] = Value;
      ++Pos;
    }
  };

  class Test {
    int a, b, c;
  public:
    constexpr Test(int *Arr, int &Pos) :
      a(BitSetter(Arr, Pos, 1).getValue()),
      b(BitSetter(Arr, Pos, 2).getValue()),
      c(BitSetter(Arr, Pos, 3).getValue())
    {}
  };


  constexpr int T(int Index) {
    int Arr[] = {0, 0, 0};
    int Pos = 0;

    {
      auto T = Test(Arr, Pos);
      // End of scope, should destroy Test.
    }

    return Arr[Index];
  }

  static_assert(T(0) == 1);
  static_assert(T(1) == 2);
  static_assert(T(2) == 3);
#endif
}

#if __cplusplus >= 201703L
namespace BaseInit {
  class _A {public: int a;};
  class _B : public _A {};
  class _C : public _B {};

  constexpr _C c{12};
  constexpr const _B &b = c;
  static_assert(b.a == 12);

  class A {public: int a;};
  class B : public A {};
  class C : public A {};
  class D : public B, public C {};

  // This initializes D::B::A::a and not D::C::A::a.
  constexpr D d{12};
  static_assert(d.B::a == 12);
  static_assert(d.C::a == 0);
};
#endif

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

  constexpr const A *aPointer = &c;
  constexpr const B *bPointer = &c;

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
#if __cplusplus < 202002L
  struct Base { // ref-note 2{{declared here}} expected-note {{declared here}}
    int Val;
  };

  struct Derived : Base {
    int OtherVal;

    constexpr Derived(int i) : OtherVal(i) {} // ref-error {{never produces a constant expression}} \
                                              // ref-note 2{{non-constexpr constructor 'Base' cannot be used in a constant expression}} \
                                              // expected-note {{non-constexpr constructor 'Base' cannot be used in a constant expression}}
  };

  constexpr Derived D(12); // ref-error {{must be initialized by a constant expression}} \
                           // ref-note {{in call to 'Derived(12)'}} \
                           // ref-note {{declared here}} \
                           // expected-error {{must be initialized by a constant expression}} \
                           // expected-note {{in call to 'Derived(12)'}}

  static_assert(D.Val == 0, ""); // ref-error {{not an integral constant expression}} \
                                 // ref-note {{initializer of 'D' is not a constant expression}} \
                                 // expected-error {{not an integral constant expression}} \
                                 // expected-note {{read of uninitialized object}}
#endif

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
    using YetAnotherBase::YetAnotherBase; // ref-note {{declared here}} \
                                          // expected-note {{declared here}}
    int OtherVal;

    constexpr bool doit() const { return Val == OtherVal; }
  };

  constexpr YetAnotherDerived Oops(0); // ref-error {{must be initialized by a constant expression}} \
                                       // ref-note {{constructor inherited from base class 'YetAnotherBase' cannot be used in a constant expression}} \
                                       // expected-error {{must be initialized by a constant expression}} \
                                       // expected-note {{constructor inherited from base class 'YetAnotherBase' cannot be used in a constant expression}}
};

namespace EmptyCtor {
  struct piecewise_construct_t { explicit piecewise_construct_t() = default; };
  constexpr piecewise_construct_t piecewise_construct =
    piecewise_construct_t();
};

namespace ConditionalInit {
  struct S { int a; };

  constexpr S getS(bool b) {
    return b ? S{12} : S{13};
  }

  static_assert(getS(true).a == 12, "");
  static_assert(getS(false).a == 13, "");
};
/// FIXME: The following tests are broken.
///   They are using CXXDefaultInitExprs which contain a CXXThisExpr. The This pointer
///   in those refers to the declaration we are currently initializing, *not* the
///   This pointer of the current stack frame. This is something we haven't
///   implemented in the new interpreter yet.
namespace DeclRefs {
  struct A{ int m; const int &f = m; }; // expected-note {{implicit use of 'this'}}

  constexpr A a{10}; // expected-error {{must be initialized by a constant expression}}
  static_assert(a.m == 10, "");
  static_assert(a.f == 10, ""); // expected-error {{not an integral constant expression}} \
                                // expected-note {{read of uninitialized object}}

  class Foo {
  public:
    int z = 1337;
    constexpr int a() const {
      A b{this->z};

      return b.f;
    }
  };
  constexpr Foo f;
  static_assert(f.a() == 1337, "");


  struct B {
    A a = A{100};
  };
  constexpr B b;
  /// FIXME: The following two lines don't work because we don't get the
  ///   pointers on the LHS correct. They make us run into an assertion
  ///   in CheckEvaluationResult. However, this may just be caused by the
  ///   problems in the previous examples.
  //static_assert(b.a.m == 100, "");
  //static_assert(b.a.f == 100, "");
}

#if __cplusplus >= 202002L
namespace VirtualCalls {
namespace Obvious {

  class A {
  public:
    constexpr A(){}
    constexpr virtual int foo() {
      return 3;
    }
  };
  class B : public A {
  public:
    constexpr int foo() override {
      return 6;
    }
  };

  constexpr int getFooB(bool b) {
    A *a;
    A myA;
    B myB;

    if (b)
      a = &myA;
    else
      a = &myB;

    return a->foo();
  }
  static_assert(getFooB(true) == 3, "");
  static_assert(getFooB(false) == 6, "");
}

namespace MultipleBases {
  class A {
  public:
    constexpr virtual int getInt() const { return 10; }
  };
  class B {
  public:
  };
  class C : public A, public B {
  public:
    constexpr int getInt() const override { return 20; }
  };

  constexpr int callGetInt(const A& a) { return a.getInt(); }
  static_assert(callGetInt(C()) == 20, "");
  static_assert(callGetInt(A()) == 10, "");
}

namespace Destructors {
  class Base {
  public:
    int i;
    constexpr Base(int &i) : i(i) {i++;}
    constexpr virtual ~Base() {i--;}
  };

  class Derived : public Base {
  public:
    constexpr Derived(int &i) : Base(i) {}
    constexpr virtual ~Derived() {i--;}
  };

  constexpr int test() {
    int i = 0;
    Derived d(i);
    return i;
  }
  static_assert(test() == 1);
}


namespace VirtualDtors {
  class A {
  public:
    unsigned &v;
    constexpr A(unsigned &v) : v(v) {}
    constexpr virtual ~A() {
      v |= (1 << 0);
    }
  };
  class B : public A {
  public:
    constexpr B(unsigned &v) : A(v) {}
    constexpr virtual ~B() {
      v |= (1 << 1);
    }
  };
  class C : public B {
  public:
    constexpr C(unsigned &v) : B(v) {}
    constexpr virtual ~C() {
      v |= (1 << 2);
    }
  };

  constexpr bool foo() {
    unsigned a = 0;
    {
      C c(a);
    }
    return ((a & (1 << 0)) && (a & (1 << 1)) && (a & (1 << 2)));
  }

  static_assert(foo());


};

namespace QualifiedCalls {
  class A {
      public:
      constexpr virtual int foo() const {
          return 5;
      }
  };
  class B : public A {};
  class C : public B {
      public:
      constexpr int foo() const override {
          return B::foo(); // B doesn't have a foo(), so this should call A::foo().
      }
      constexpr int foo2() const {
        return this->A::foo();
      }
  };
  constexpr C c;
  static_assert(c.foo() == 5);
  static_assert(c.foo2() == 5);


  struct S {
    int _c = 0;
    virtual constexpr int foo() const { return 1; }
  };

  struct SS : S {
    int a;
    constexpr SS() {
      a = S::foo();
    }
    constexpr int foo() const override {
      return S::foo();
    }
  };

  constexpr SS ss;
  static_assert(ss.a == 1);
}

namespace CtorDtor {
  struct Base {
    int i = 0;
    int j = 0;

    constexpr Base() : i(func()) {
      j = func();
    }
    constexpr Base(int i) : i(i), j(i) {}

    constexpr virtual int func() const { return 1; }
  };

  struct Derived : Base {
    constexpr Derived() {}
    constexpr Derived(int i) : Base(i) {}
    constexpr int func() const override { return 2; }
  };

  struct Derived2 : Derived {
    constexpr Derived2() : Derived(func()) {} // ref-note {{subexpression not valid in a constant expression}}
    constexpr int func() const override { return 3; }
  };

  constexpr Base B;
  static_assert(B.i == 1 && B.j == 1, "");

  constexpr Derived D;
  static_assert(D.i == 1, ""); // expected-error {{static assertion failed}} \
                               // expected-note {{2 == 1}}
  static_assert(D.j == 1, ""); // expected-error {{static assertion failed}} \
                               // expected-note {{2 == 1}}

  constexpr Derived2 D2; // ref-error {{must be initialized by a constant expression}} \
                         // ref-note {{in call to 'Derived2()'}} \
                         // ref-note 2{{declared here}}
  static_assert(D2.i == 3, ""); // ref-error {{not an integral constant expression}} \
                                // ref-note {{initializer of 'D2' is not a constant expression}}
  static_assert(D2.j == 3, ""); // ref-error {{not an integral constant expression}} \
                                // ref-note {{initializer of 'D2' is not a constant expression}}

}
};
#endif
