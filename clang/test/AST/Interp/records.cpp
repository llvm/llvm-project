// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++14 -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple i686 -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s
// RUN: %clang_cc1 -verify=ref,both -std=c++14 %s
// RUN: %clang_cc1 -verify=ref,both -std=c++20 %s
// RUN: %clang_cc1 -verify=ref,both -triple i686 %s

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
constexpr Ints2 ints22; // both-error {{without a user-provided default constructor}} \
                        // expected-error {{must be initialized by a constant expression}}

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
    S{}; // ref-note {{in call to 'S{}.~S()'}} \
         // expected-note {{in call to '&S{}->~S()'}}
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
/// FIXME: The following tests are broken.
///   They are using CXXDefaultInitExprs which contain a CXXThisExpr. The This pointer
///   in those refers to the declaration we are currently initializing, *not* the
///   This pointer of the current stack frame. This is something we haven't
///   implemented in the new interpreter yet.
namespace DeclRefs {
  struct A{ int m; const int &f = m; }; // expected-note {{implicit use of 'this'}}

  constexpr A a{10}; // expected-error {{must be initialized by a constant expression}} \
                     // expected-note {{declared here}}
  static_assert(a.m == 10, "");
  static_assert(a.f == 10, ""); // expected-error {{not an integral constant expression}} \
                                // expected-note {{initializer of 'a' is not a constant expression}}

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
    S{}; // ref-note {{in call to 'S{}.~S()'}} \
         // expected-note {{in call to '&S{}->~S()'}}
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

  /// FIXME: This needs support for unions on the new interpreter.
  /// We diagnose an uninitialized object in c++14.
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
      return f.i + f.U.f; // ref-note {{read of member 'f' of union with active member 'a'}}
    }
    static_assert(foo(F()) == 0, ""); // ref-error {{not an integral constant expression}} \
                                      // ref-note {{in call to}}
  }
#endif

#if __cplusplus >= 202002L
  namespace Failure {
    struct S {
      int a;
      F f{12};
    };
    constexpr int foo(S x) {
      return x.a; // expected-note {{read of uninitialized object}}
    }
    static_assert(foo(S()) == 0, ""); // expected-error {{not an integral constant expression}} \
                                      // expected-note {{in call to}}
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
  O o1(0);
  constinit O o2(0); // both-error {{variable does not have a constant initializer}} \
                     // both-note {{required by 'constinit' specifier}} \
                     // both-note {{reference to temporary is not a constant expression}} \
                     // both-note {{temporary created here}}
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
