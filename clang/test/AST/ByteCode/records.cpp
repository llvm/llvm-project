// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++14 -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++17 -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++17 -triple i686 -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both -std=c++14 %s
// RUN: %clang_cc1 -verify=ref,both -std=c++17 %s
// RUN: %clang_cc1 -verify=ref,both -std=c++17 -triple i686 %s
// RUN: %clang_cc1 -verify=ref,both -std=c++20 %s

/// Used to crash.
struct Empty {};
constexpr Empty e = {Empty()};

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
constexpr Ints2 ints22; // both-error {{without a user-provided default constructor}}

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


/// A global, composite temporary variable.
constexpr const C &c3 = C().get();

/// Same, but with a bitfield.
class D {
public:
  unsigned a : 4;
  constexpr D() : a(15) {}
  constexpr D get() const {
    return *this;
  }
};
constexpr const D &d4 = D().get();

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

class Bar { // both-note {{definition of 'Bar' is not complete}}
public:
  constexpr Bar(){}
  constexpr Bar b; // both-error {{cannot be constexpr}} \
                   // both-error {{has incomplete type 'const Bar'}}
};
constexpr Bar B; // both-error {{must be initialized by a constant expression}}
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

  constexpr int foo() { // both-error {{never produces a constant expression}}
    S *s = nullptr;
    return s->get12(); // both-note 2{{member call on dereferenced null pointer}}

  }
  static_assert(foo() == 12, ""); // both-error {{not an integral constant expression}} \
                                  // both-note {{in call to 'foo()'}}
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
static_assert(a2.i == 200, ""); // both-error {{static assertion failed}} \
                                // both-note {{evaluates to '12 == 200'}}


struct S {
  int a = 0;
  constexpr int get5() const { return 5; }
  constexpr void fo() const {
    this; // both-warning {{expression result unused}}
    this->a; // both-warning {{expression result unused}}
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
      Test(Arr, Pos);
      // End of scope, should destroy Test.
    }

    return Arr[Index];
  }
  static_assert(T(0) == 1);
  static_assert(T(1) == 2);
  static_assert(T(2) == 3);

  // Invalid destructor.
  struct S {
    constexpr S() {}
    constexpr ~S() noexcept(false) { throw 12; } // both-error {{cannot use 'throw'}} \
                                                 // both-error {{never produces a constant expression}} \
                                                 // both-note 2{{subexpression not valid}}
  };

  constexpr int f() {
    S{}; // both-note {{in call to 'S{}.~S()'}}
    return 12;
  }
  static_assert(f() == 12); // both-error {{not an integral constant expression}} \
                            // both-note {{in call to 'f()'}}


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
  struct Base { // both-note {{declared here}} \
                // ref-note {{declared here}}
    int Val;
  };

  struct Derived : Base {
    int OtherVal;

    constexpr Derived(int i) : OtherVal(i) {} // ref-error {{never produces a constant expression}} \
                                              // both-note {{non-constexpr constructor 'Base' cannot be used in a constant expression}} \
                                              // ref-note {{non-constexpr constructor 'Base' cannot be used in a constant expression}}
  };

  constexpr Derived D(12); // both-error {{must be initialized by a constant expression}} \
                           // both-note {{in call to 'Derived(12)'}} \
                           // both-note {{declared here}}

  static_assert(D.Val == 0, ""); // both-error {{not an integral constant expression}} \
                                 // both-note {{initializer of 'D' is not a constant expression}}
#endif

  struct AnotherBase {
    int Val;
    constexpr AnotherBase(int i) : Val(12 / i) {} // both-note {{division by zero}}
  };

  struct AnotherDerived : AnotherBase {
    constexpr AnotherDerived(int i) : AnotherBase(i) {}
  };
  constexpr AnotherBase Derp(0); // both-error {{must be initialized by a constant expression}} \
                                 // both-note {{in call to 'AnotherBase(0)'}}

  struct YetAnotherBase {
    int Val;
    constexpr YetAnotherBase(int i) : Val(i) {}
  };

  struct YetAnotherDerived : YetAnotherBase {
    using YetAnotherBase::YetAnotherBase; // both-note {{declared here}}
    int OtherVal;

    constexpr bool doit() const { return Val == OtherVal; }
  };

  constexpr YetAnotherDerived Oops(0); // both-error {{must be initialized by a constant expression}} \
                                       // both-note {{constructor inherited from base class 'YetAnotherBase' cannot be used in a constant expression}}
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
namespace DeclRefs {
  struct A{ int m; const int &f = m; };

  constexpr A a{10};
  static_assert(a.m == 10, "");
  static_assert(a.f == 10, "");

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
  static_assert(b.a.m == 100, "");
  static_assert(b.a.f == 100, "");

  constexpr B b2{};
  static_assert(b2.a.m == 100, "");
  static_assert(b2.a.f == 100, "");
  static_assert(b2.a.f == 101, ""); // both-error {{failed}} \
                                    // both-note {{evaluates to '100 == 101'}}
}

namespace PointerArith {
  struct A {};
  struct B : A { int n; };

  B b = {};
  constexpr A *a1 = &b;
  constexpr B *b1 = &b + 1;
  constexpr B *b2 = &b + 0;

  constexpr A *a2 = &b + 1; // both-error {{must be initialized by a constant expression}} \
                            // both-note {{cannot access base class of pointer past the end of object}}
  constexpr const int *pn = &(&b + 1)->n; // both-error {{must be initialized by a constant expression}} \
                                          // both-note {{cannot access field of pointer past the end of object}}
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

  struct S {
    constexpr S() {}
    constexpr ~S() { // both-error {{never produces a constant expression}}
      int i = 1 / 0; // both-warning {{division by zero}} \
                     // both-note 2{{division by zero}}
    }
  };
  constexpr int testS() {
    S{}; // both-note {{in call to 'S{}.~S()'}}
    return 1;
  }
  static_assert(testS() == 1); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to 'testS()'}}
}

namespace BaseToDerived {
namespace A {
  struct A {};
  struct B : A { int n; };
  struct C : B {};
  C c = {};
  constexpr C *pb = (C*)((A*)&c + 1); // both-error {{must be initialized by a constant expression}} \
                                      // both-note {{cannot access derived class of pointer past the end of object}}
}
namespace B {
  struct A {};
  struct Z {};
  struct B : Z, A {
    int n;
   constexpr B() : n(10) {}
  };
  struct C : B {
   constexpr C() : B() {}
  };

  constexpr C c = {};
  constexpr const A *pa = &c;
  constexpr const C *cp = (C*)pa;
  constexpr const B *cb = (B*)cp;

  static_assert(cb->n == 10);
  static_assert(cp->n == 10);
}

namespace C {
  struct Base { int *a; };
  struct Base2 : Base { int f[12]; };

  struct Middle1 { int b[3]; };
  struct Middle2 : Base2 { char c; };
  struct Middle3 : Middle2 { char g[3]; };
  struct Middle4 { int f[3]; };
  struct Middle5 : Middle4, Middle3 { char g2[3]; };

  struct NotQuiteDerived : Middle1, Middle5 { bool d; };
  struct Derived : NotQuiteDerived { int e; };

  constexpr NotQuiteDerived NQD1 = {};

  constexpr Middle5 *M4 = (Middle5*)((Base2*)&NQD1);
  static_assert(M4->a == nullptr);
  static_assert(M4->g2[0] == 0);
}
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

namespace VirtualFunctionPointers {
  struct S {
    virtual constexpr int func() const { return 1; }
  };

  struct Middle : S {
    constexpr Middle(int i) : i(i) {}
    int i;
  };

  struct Other {
    constexpr Other(int k) : k(k) {}
    int k;
  };

  struct S2 : Middle, Other {
    int j;
    constexpr S2(int i, int j, int k) : Middle(i), Other(k), j(j) {}
    virtual constexpr int func() const { return i + j + k  + S::func(); }
  };

  constexpr S s;
  constexpr decltype(&S::func) foo = &S::func;
  constexpr int value = (s.*foo)();
  static_assert(value == 1);


  constexpr S2 s2(1, 2, 3);
  static_assert(s2.i == 1);
  static_assert(s2.j == 2);
  static_assert(s2.k == 3);

  constexpr int value2 = s2.func();
  constexpr int value3 = (s2.*foo)();
  static_assert(value3 == 7);

  constexpr int dynamicDispatch(const S &s) {
    constexpr decltype(&S::func) SFunc = &S::func;

    return (s.*SFunc)();
  }

  static_assert(dynamicDispatch(s) == 1);
  static_assert(dynamicDispatch(s2) == 7);
};

};
#endif

#if __cplusplus < 202002L
namespace VirtualFromBase {
  struct S1 {
    virtual int f() const;
  };
  struct S2 {
    virtual int f();
  };
  template <typename T> struct X : T {
    constexpr X() {}
    double d = 0.0;
    constexpr int f() { return sizeof(T); }
  };

  // Non-virtual f(), OK.
  constexpr X<X<S1>> xxs1;
  constexpr X<S1> *p = const_cast<X<X<S1>>*>(&xxs1);
  static_assert(p->f() == sizeof(S1), "");

  // Virtual f(), not OK.
  constexpr X<X<S2>> xxs2;
  constexpr X<S2> *q = const_cast<X<X<S2>>*>(&xxs2);
  static_assert(q->f() == sizeof(X<S2>), ""); // both-error {{not an integral constant expression}} \
                                              // both-note {{cannot evaluate call to virtual function}}
}
#endif

namespace CompositeDefaultArgs {
  struct Foo {
    int a;
    int b;
    constexpr Foo() : a(12), b(13) {}
  };

  class Bar {
  public:
    bool B = false;

    constexpr int someFunc(Foo F = Foo()) {
      this->B = true;
      return 5;
    }
  };

  constexpr bool testMe() {
    Bar B;
    B.someFunc();
    return B.B;
  }
  static_assert(testMe(), "");
}

constexpr bool BPand(BoolPair bp) {
  return bp.first && bp.second;
}
static_assert(BPand(BoolPair{true, false}) == false, "");

namespace TemporaryObjectExpr {
  struct F {
    int a;
    constexpr F() : a(12) {}
  };
  constexpr int foo(F f) {
    return 0;
  }
  static_assert(foo(F()) == 0, "");
}

  namespace ZeroInit {
  struct F {
    int a;
  };

  namespace Simple {
    struct A {
      char a;
      bool b;
      int c[4];
      float d;
    };
    constexpr int foo(A x) {
      return x.a + static_cast<int>(x.b) + x.c[0] + x.c[3] + static_cast<int>(x.d);
    }
    static_assert(foo(A()) == 0, "");
  }

  namespace Inheritance {
    struct F2 : F {
      float f;
    };

    constexpr int foo(F2 f) {
      return (int)f.f + f.a;
    }
    static_assert(foo(F2()) == 0, "");
  }

  namespace BitFields {
    struct F {
      unsigned a : 6;
    };
    constexpr int foo(F f) {
      return f.a;
    }
    static_assert(foo(F()) == 0, "");
  }

  namespace Nested {
    struct F2 {
      float f;
      char c;
    };

    struct F {
      F2 f2;
      int i;
    };

    constexpr int foo(F f) {
      return f.i + f.f2.f + f.f2.c;
    }
    static_assert(foo(F()) == 0, "");
  }

  namespace CompositeArrays {
    struct F2 {
      float f;
      char c;
    };

    struct F {
      F2 f2[2];
      int i;
    };

    constexpr int foo(F f) {
      return f.i + f.f2[0].f + f.f2[0].c + f.f2[1].f + f.f2[1].c;
    }
    static_assert(foo(F()) == 0, "");
  }

#if __cplusplus > 201402L
  namespace Unions {
    struct F {
      union {
        int a;
        char c[4];
        float f;
      } U;
      int i;
    };

    constexpr int foo(F f) {
      return f.i + f.U.f; // both-note {{read of member 'f' of union with active member 'a'}}
    }
    static_assert(foo(F()) == 0, ""); // both-error {{not an integral constant expression}} \
                                      // both-note {{in call to}}
  }
#endif

#if __cplusplus >= 202002L
  namespace Failure {
    struct S {
      int a;
      F f{12};
    };
    constexpr int foo(S x) {
      return x.a;
    }
    static_assert(foo(S()) == 0, "");
  };
#endif
}

#if __cplusplus >= 202002L
namespace ParenInit {
  struct A {
    int a;
  };

  struct B : A {
    int b;
  };

  constexpr B b(A(1),2);


  struct O {
    int &&j;
  };

  /// Not constexpr!
  O o1(0); // both-warning {{temporary whose address is used as value of}}
  // FIXME: the secondary warning message is bogus, would be nice to suppress it.
  constinit O o2(0); // both-error {{variable does not have a constant initializer}} \
                     // both-note {{required by 'constinit' specifier}} \
                     // both-note {{reference to temporary is not a constant expression}} \
                     // both-note {{temporary created here}} \
                     // both-warning {{temporary whose address is used as value}}


  /// Initializing an array.
  constexpr void bar(int i, int j) {
    int arr[4](i, j);
  }
}
#endif

namespace DelegatingConstructors {
  struct S {
    int a;
    constexpr S() : S(10) {}
    constexpr S(int a) : a(a) {}
  };
  constexpr S s = {};
  static_assert(s.a == 10, "");

  struct B {
    int a;
    int b;

    constexpr B(int a) : a(a), b(a + 2) {}
  };
  struct A : B {
    constexpr A() : B(10) {};
  };
  constexpr A d4 = {};
  static_assert(d4.a == 10, "");
  static_assert(d4.b == 12, "");
}

namespace AccessOnNullptr {
  struct F {
    int a;
  };

  constexpr int a() { // both-error {{never produces a constant expression}}
    F *f = nullptr;

    f->a = 0; // both-note 2{{cannot access field of null pointer}}
    return f->a;
  }
  static_assert(a() == 0, ""); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to 'a()'}}

  constexpr int a2() { // both-error {{never produces a constant expression}}
    F *f = nullptr;


    const int *a = &(f->a); // both-note 2{{cannot access field of null pointer}}
    return f->a;
  }
  static_assert(a2() == 0, ""); // both-error {{not an integral constant expression}} \
                                // both-note {{in call to 'a2()'}}
}

namespace IndirectFieldInit {
#if __cplusplus >= 202002L
  /// Primitive.
  struct Nested1 {
    struct {
      int first;
    };
    int x;
    constexpr Nested1(int x) : first(12), x() { x = 4; }
    constexpr Nested1() : Nested1(42) {}
  };
  constexpr Nested1 N1{};
  static_assert(N1.first == 12, "");

  /// Composite.
  struct Nested2 {
    struct First { int x = 42; };
    struct {
      First first;
    };
    int x;
    constexpr Nested2(int x) : first(12), x() { x = 4; }
    constexpr Nested2() : Nested2(42) {}
  };
  constexpr Nested2 N2{};
  static_assert(N2.first.x == 12, "");

  /// Bitfield.
  struct Nested3 {
    struct {
      unsigned first : 2;
    };
    int x;
    constexpr Nested3(int x) : first(3), x() { x = 4; }
    constexpr Nested3() : Nested3(42) {}
  };

  constexpr Nested3 N3{};
  static_assert(N3.first == 3, "");

  /// Test that we get the offset right if the
  /// record has a base.
  struct Nested4Base {
    int a;
    int b;
    char c;
  };
  struct Nested4 : Nested4Base{
    struct {
      int first;
    };
    int x;
    constexpr Nested4(int x) : first(123), x() { a = 1; b = 2; c = 3; x = 4; }
    constexpr Nested4() : Nested4(42) {}
  };
  constexpr Nested4 N4{};
  static_assert(N4.first == 123, "");

  struct S {
    struct {
      int x, y;
    };

    constexpr S(int x_, int y_) : x(x_), y(y_) {}
  };

  constexpr S s(1, 2);
  static_assert(s.x == 1 && s.y == 2);

  struct S2 {
    int a;
    struct {
      int b;
      struct {
        int x, y;
      };
    };

    constexpr S2(int x_, int y_) : a(3), b(4), x(x_), y(y_) {}
  };

  constexpr S2 s2(1, 2);
  static_assert(s2.x == 1 && s2.y == 2 && s2.a == 3 && s2.b == 4);

#endif
}

namespace InheritedConstructor {
  namespace PR47555 {
    struct A {
      int c;
      int d;
      constexpr A(int c, int d) : c(c), d(d){}
    };
    struct B : A { using A::A; };

    constexpr B b = {13, 1};
    static_assert(b.c == 13, "");
    static_assert(b.d == 1, "");
  }

  namespace PR47555_2 {
    struct A {
      int c;
      int d;
      double e;
      constexpr A(int c, int &d, double e) : c(c), d(++d), e(e){}
    };
    struct B : A { using A::A; };

    constexpr int f() {
      int a = 10;
      B b = {10, a, 40.0};
      return a;
    }
    static_assert(f() == 11, "");
  }

  namespace AaronsTest {
    struct T {
      constexpr T(float) {}
    };

    struct Base {
      constexpr Base(T t = 1.0f) {}
      constexpr Base(float) {}
    };

    struct FirstMiddle : Base {
      using Base::Base;
      constexpr FirstMiddle() : Base(2.0f) {}
    };

    struct SecondMiddle : Base {
      constexpr SecondMiddle() : Base(3.0f) {}
      constexpr SecondMiddle(T t) : Base(t) {}
    };

    struct S : FirstMiddle, SecondMiddle {
      using FirstMiddle::FirstMiddle;
      constexpr S(int i) : S(4.0f) {}
    };

    constexpr S s(1);
  }
}

namespace InvalidCtorInitializer {
  struct X {
    int Y;
    constexpr X()
        : Y(fo_o_()) {} // both-error {{use of undeclared identifier 'fo_o_'}}
  };
  // no crash on evaluating the constexpr ctor.
  constexpr int Z = X().Y; // both-error {{constexpr variable 'Z' must be initialized by a constant expression}}
}

extern int f(); // both-note {{here}}
struct HasNonConstExprMemInit {
  int x = f(); // both-note {{non-constexpr function}}
  constexpr HasNonConstExprMemInit() {} // both-error {{never produces a constant expression}}
};

namespace {
  template <class Tp, Tp v>
  struct integral_constant {
    static const Tp value = v;
  };

  template <class Tp, Tp v>
  const Tp integral_constant<Tp, v>::value;

  typedef integral_constant<bool, true> true_type;
  typedef integral_constant<bool, false> false_type;

  /// This might look innocent, but we get an evaluateAsInitializer call for the
  /// static bool member before evaluating the first static_assert, but we do NOT
  /// get such a call for the second one. So the second one needs to lazily visit
  /// the data member itself.
  static_assert(true_type::value, "");
  static_assert(true_type::value, "");
}

#if __cplusplus >= 202002L
namespace {
  /// Used to crash because the CXXDefaultInitExpr is of compound type.
  struct A {
    int &x;
    constexpr ~A() { --x; }
  };
  struct B {
    int &x;
    const A &a = A{x};
  };
  constexpr int a() {
    int x = 1;
    int f = B{x}.x;
    B{x}; // both-warning {{expression result unused}}

    return 1;
  }
}
#endif

namespace pr18633 {
  struct A1 {
    static const int sz;
    static const int sz2;
  };
  const int A1::sz2 = 11;
  template<typename T>
  void func () {
    int arr[A1::sz];
    // both-warning@-1 {{variable length arrays in C++ are a Clang extension}}
    // both-note@-2 {{initializer of 'sz' is unknown}}
    // both-note@-9 {{declared here}}
  }
  template<typename T>
  void func2 () {
    int arr[A1::sz2];
  }
  const int A1::sz = 12;
  void func2() {
    func<int>();
    func2<int>();
  }
}

namespace {
  struct F {
    static constexpr int Z = 12;
  };
  F f;
  static_assert(f.Z == 12, "");
}

namespace UnnamedBitFields {
  struct A {
    int : 1;
    double f;
    int : 1;
    char c;
  };

  constexpr A a = (A){1.0, 'a'};
  static_assert(a.f == 1.0, "");
  static_assert(a.c == 'a', "");
}

namespace VirtualBases {
  /// This used to crash.
  namespace One {
    class A {
    protected:
      int x;
    };
    class B : public virtual A {
    public:
      int getX() { return x; } // both-note {{declared here}}
    };

    class DV : virtual public B{};

    void foo() {
      DV b;
      int a[b.getX()]; // both-warning {{variable length arrays}} \
                       // both-note {{non-constexpr function 'getX' cannot be used}}
    }
  }

  namespace Two {
    struct U { int n; };
    struct A : virtual U { int n; };
    struct B : A {};
    B a;
    static_assert((U*)(A*)(&a) == (U*)(&a), "");

    struct C : virtual A {};
    struct D : B, C {};
    D d;
    constexpr B *p = &d;
    constexpr C *q = &d;
    static_assert((A*)p == (A*)q, ""); // both-error {{failed}}
  }

  namespace Three {
    struct U { int n; };
    struct V : U { int n; };
    struct A : virtual V { int n; };
    struct Aa { int n; };
    struct B : virtual A, Aa {};

    struct C : virtual A, Aa {};

    struct D : B, C {};

    D d;

    constexpr B *p = &d;
    constexpr C *q = &d;

    static_assert((void*)p != (void*)q, "");
    static_assert((A*)p == (A*)q, "");
    static_assert((Aa*)p != (Aa*)q, "");

    constexpr V *v = p;
    constexpr V *w = q;
    constexpr V *x = (A*)p;
    static_assert(v == w, "");
    static_assert(v == x, "");

    static_assert((U*)&d == p, "");
    static_assert((U*)&d == q, "");
    static_assert((U*)&d == v, "");
    static_assert((U*)&d == w, "");
    static_assert((U*)&d == x, "");

    struct X {};
    struct Y1 : virtual X {};
    struct Y2 : X {};
    struct Z : Y1, Y2 {};
    Z z;
    static_assert((X*)(Y1*)&z != (X*)(Y2*)&z, "");
  }
}

namespace ZeroInit {
  struct S3 {
    S3() = default;
    S3(const S3&) = default;
    S3(S3&&) = default;
    constexpr S3(int n) : n(n) {}
    int n;
  };
  constexpr S3 s3d; // both-error {{default initialization of an object of const type 'const S3' without a user-provided default constructor}}
  static_assert(s3d.n == 0, "");

  struct P {
    int a = 10;
  };
  static_assert(P().a == 10, "");
}

namespace {
#if __cplusplus >= 202002L
  struct C {
    template <unsigned N> constexpr C(const char (&)[N]) : n(N) {}
    unsigned n;
  };
  template <C c>
  constexpr auto operator""_c() { return c.n; }

  constexpr auto waldo = "abc"_c;
  static_assert(waldo == 4, "");
#endif
}


namespace TemporaryWithInvalidDestructor {
#if __cplusplus >= 202002L
  struct A {
    bool a = true;
    constexpr ~A() noexcept(false) { // both-error {{never produces a constant expression}}
      throw; // both-note 2{{not valid in a constant expression}} \
             // both-error {{cannot use 'throw' with exceptions disabled}}
    }
  };
  static_assert(A().a, ""); // both-error {{not an integral constant expression}} \
                        // both-note {{in call to}}
#endif
}

namespace IgnoredCtorWithZeroInit {
  struct S {
    int a;
  };

  bool get_status() {
    return (S(), true);
  }
}

#if __cplusplus >= 202002L
namespace VirtOperator {
  /// This used to crash because it's a virtual CXXOperatorCallExpr.
  struct B {
    virtual constexpr bool operator==(const B&) const { return true; }
  };
  struct D : B {
    constexpr bool operator==(const B&) const override{ return false; } // both-note {{operator}}
  };
  constexpr bool cmp_base_derived = D() == D(); // both-warning {{ambiguous}}
}

namespace FloatAPValue {
  struct ClassTemplateArg {
    int a;
    float f;
  };
  template<ClassTemplateArg A> struct ClassTemplateArgTemplate {
    static constexpr const ClassTemplateArg &Arg = A;
  };
  ClassTemplateArgTemplate<ClassTemplateArg{1, 2.0f}> ClassTemplateArgObj;
  template<const ClassTemplateArg&> struct ClassTemplateArgRefTemplate {};
  ClassTemplateArgRefTemplate<ClassTemplateArgObj.Arg> ClassTemplateArgRefObj;
}
#endif

namespace LocalWithThisPtrInit {
  struct S {
    int i;
    int *p = &i;
  };
  constexpr int foo() {
    S s{2};
    return *s.p;
  }
  static_assert(foo() == 2, "");
}

namespace OnePastEndAndBack {
  struct Base {
    constexpr Base() {}
    int n = 0;
  };

  constexpr Base a;
  constexpr const Base *c = &a + 1;
  constexpr const Base *d = c - 1;
  static_assert(d == &a, "");
}

namespace BitSet {
  class Bitset {
    unsigned Bit = 0;

  public:
    constexpr Bitset() {
      int Init[2] = {1,2};
      for (auto I : Init)
        set(I);
    }
    constexpr void set(unsigned I) {
      this->Bit++;
      this->Bit = 1u << 1;
    }
  };

  struct ArchInfo {
    Bitset DefaultExts;
  };

  constexpr ArchInfo ARMV8A = {
    Bitset()
  };
}

namespace ArrayInitChain {
  struct StringLiteral {
    const char *S;
  };

  struct CustomOperandVal {
    StringLiteral Str;
    unsigned Width;
    unsigned Mask = Width + 1;
  };

  constexpr CustomOperandVal A[] = {
    {},
    {{"depctr_hold_cnt"},  12,   13},
  };
  static_assert(A[0].Str.S == nullptr, "");
  static_assert(A[0].Width == 0, "");
  static_assert(A[0].Mask == 1, "");

  static_assert(A[1].Width == 12, "");
  static_assert(A[1].Mask == 13, "");
}

#if __cplusplus >= 202002L
namespace ctorOverrider {
  // Ensure that we pick the right final overrider during construction.
  struct A {
    virtual constexpr char f() const { return 'A'; }
    char a = f();
  };

  struct Covariant1 {
    A d;
  };

  constexpr Covariant1 cb;
}
#endif

#if __cplusplus >= 202002L
namespace VirtDtor {
  struct X { char *p; constexpr ~X() { *p++ = 'X'; } };
  struct Y : X { int y; virtual constexpr ~Y() { *p++ = 'Y'; } };
  struct Z : Y { int z; constexpr ~Z() override { *p++ = 'Z'; } };

  union VU {
    constexpr VU() : z() {}
    constexpr ~VU() {}
    Z z;
  };

  constexpr char virt_dtor(int mode, const char *expected) {
    char buff[4] = {};
    VU vu;
    vu.z.p = buff;

    ((Y&)vu.z).~Y();
    return true;
  }
  static_assert(virt_dtor(0, "ZYX"));
}

namespace DtorDestroysFieldsAfterSelf {
    struct  S {
      int a = 10;
      constexpr ~S() {
        a = 0;
      }

    };
    struct F {
      S s;
      int a;
      int &b;
      constexpr F(int a, int &b) : a(a), b(b) {}
      constexpr ~F() {
        b += s.a;
      }
    };

  constexpr int foo() {
    int a = 10;
    int b = 5;
    {
      F f(a, b);
    }

    return b;
  }

  static_assert(foo() == 15);
}
#endif

namespace ExprWithCleanups {
  struct A { A(); ~A(); int get(); };
  constexpr int get() {return false ? A().get() : 1;}
  static_assert(get() == 1, "");


  struct S {
    int V;
    constexpr S(int V) : V(V) {}
    constexpr int get() {
      return V;
    }
  };
  constexpr int get(bool b) {
    S a = b ? S(1) : S(2);

    return a.get();
  }
  static_assert(get(true) == 1, "");
  static_assert(get(false) == 2, "");


  constexpr auto F = true ? 1i : 2i;
  static_assert(F == 1i, "");
}

namespace NullptrCast {
  struct A {};
  struct B : A { int n; };
  constexpr A *na = nullptr;
  constexpr B *nb = nullptr;
  constexpr A &ra = *nb; // both-error {{constant expression}} \
                         // both-note {{cannot access base class of null pointer}}
  constexpr B &rb = (B&)*na; // both-error {{constant expression}} \
                             // both-note {{cannot access derived class of null pointer}}
  constexpr bool test() {
    auto a = (A*)(B*)nullptr;

    return a == nullptr;
  }
  static_assert(test(), "");

  constexpr bool test2() {
    auto a = (B*)(A*)nullptr;

    return a == nullptr;
  }
  static_assert(test2(), "");
}

namespace NonConst {
  template <int I>
  struct S {
    static constexpr int Size = I;
    constexpr int getSize() const { return I; }
    explicit S(int a) {}
  };

  void func() {
    int a,b ;
    const S<10> s{a};
    static_assert(s.getSize() == 10, "");
  }
}

namespace ExplicitThisInTemporary {
  struct B { B *p = this; };
  constexpr bool g(B b) { return &b == b.p; }
  static_assert(g({}), "");
}

namespace IgnoredMemberExpr {
  class A {
  public:
    int a;
  };
  class B : public A {
  public:
    constexpr int foo() {
      a; // both-warning {{expression result unused}}
      return 0;
    }
  };
  static_assert(B{}.foo() == 0, "");
}

#if __cplusplus >= 202002L
namespace DeadUpcast {
  struct A {};
  struct B : A{};
  constexpr bool foo() {

    B *pb;
    {
      B b;
      pb = &b;
    }
    A *pa = pb;

    return true;
  }
  static_assert(foo(), "");
}
#endif

namespace CtorOfInvalidClass {
  constexpr struct { Unknown U; } InvalidCtor; // both-error {{unknown type name 'Unknown'}} \
                                               // both-error {{must be initialized by a constant expression}}

#if __cplusplus >= 202002L
  template <typename T, auto Q>// both-note {{template parameter is declared here}}
  concept ReferenceOf = Q;
  /// This calls a valid and constexpr copy constructor of InvalidCtor,
  /// but should still be rejected.
  template<ReferenceOf<InvalidCtor> auto R, typename Rep> int F; // both-error {{non-type template argument is not a constant expression}}
#endif
}

namespace IncompleteTypes {
  struct Incomplete;

  constexpr bool foo() {
    extern Incomplete bounded[10];
    extern Incomplete unbounded[];
    extern Incomplete IT;
    return true;
  }
  static_assert(foo(), "");
}
