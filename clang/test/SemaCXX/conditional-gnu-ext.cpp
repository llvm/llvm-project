// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify=expected,expected-cxx11 -std=c++11 -Wsign-conversion %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify=expected,expected-cxx11 -std=c++11 -Wsign-conversion %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify=expected,expected-cxx17 -std=c++17 -Wsign-conversion %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify=expected,expected-cxx17 -std=c++17 -Wsign-conversion %s -fexperimental-new-constant-interpreter

// FIXME: Fix 'FIXME' in `Sema::ActOnConditionalOp` in 'clang/lib/Sema/SemaExpr.cpp'
// then remove `#if 0`

struct B;
struct A {
  A();
  A(const B&); // expected-note {{candidate constructor}}
  explicit operator bool();
};
struct B {
  operator A() const; // expected-note {{candidate function}}
  explicit operator bool();
};
struct I { operator bool(); };
struct J {
  operator I();
  explicit operator bool();
};
struct K { operator double(); };
typedef void (*vfn)();
struct F {
  operator vfn();
  explicit operator bool();
};
struct G { operator vfn(); };

struct Base {
  bool trick();
  A trick() const;
  void fn1();
  explicit operator bool() const;
};
struct Derived : Base {
  void fn2();
  explicit operator bool() const;
};
struct Convertible {
  operator Base&();
  explicit operator bool();
};
struct Priv : private Base { explicit operator bool(); }; // expected-note 4 {{declared private here}}
struct Mid : Base {};
struct Fin : Mid, Derived { explicit operator bool(); };
typedef void (Derived::*DFnPtr)();
struct ToMemPtr { operator DFnPtr(); };

struct BadDerived;
struct BadBase {
  operator BadDerived&();
  explicit operator bool();
};
struct BadDerived : BadBase { explicit operator bool(); };

struct Fields {
  int i1, i2, b1 : 3;
};
struct MixedFields {
  int i;
  volatile int vi;
  const int ci;
  const volatile int cvi;
};
struct MixedFieldsDerived : MixedFields {};

enum Enum { EVal };

struct Ambig {
  operator short(); // expected-note 2 {{candidate function}}
  operator signed char(); // expected-note 2 {{candidate function}}
  explicit operator bool();
};

struct Abstract {
  virtual ~Abstract() = 0;
  explicit operator bool() const;
};

struct Derived1: Abstract {};

struct Derived2: Abstract {};

#if __cplusplus >= 201402L
constexpr int evaluate_once(int x) {
  return (++x) ? : 10;
}
static_assert(evaluate_once(0) == 1, "");
#endif

void p2() {
  bool b = (bool)(A());

  // one or both void, and throwing
  b = b ? : throw 0;
  b = b ? : (throw 0);
  b ? : p2(); // expected-error {{right operand to ? is void, but left operand is of type 'bool'}}
  (b ? : throw 0) = 0;
  (b ? : (throw 0)) = 0;
  (b ? : (void)(throw 0)) = 0; // expected-error {{right operand to ? is void, but left operand is of type 'bool'}}
  bool &throwRef = (b ? : throw 0);
}

void p3() {
  // one or both class type, convert to each other
  // b1 (lvalues)
  bool b;

  Base base;
  Derived derived;
  Convertible conv;
  Base &bar1 = base ? : derived;
  Base &bar2 = derived ? : base;
  Base &bar3 = base ? : conv;
  Base &bar4 = conv ? : base;
  // these are ambiguous
  BadBase bb;
  BadDerived bd;
  (void)(bb ? : bd); // expected-error {{conditional expression is ambiguous; 'BadBase' can be converted to 'BadDerived' and vice versa}}
  (void)(bd ? : bb); // expected-error {{conditional expression is ambiguous}}
  (void)(BadBase() ? : BadDerived());
  (void)(BadDerived() ? : BadBase());

  // b2.1 (hierarchy stuff)
  extern const Base constret();
  extern const Derived constder();
  // should use const overload
  A a1((constret() ? : Base()).trick());
  A a2((Base() ? : constret()).trick());
  A a3((constret() ? : Derived()).trick());
  A a4((Derived() ? : constret()).trick());
  // should use non-const overload
  b = (Base() ? : Base()).trick();
  b = (Base() ? : Base()).trick();
  b = (Base() ? : Derived()).trick();
  b = (Derived() ? : Base()).trick();
  // should fail: const lost
  (void)(Base() ? : constder()); // expected-error {{incompatible operand types ('Base' and 'const Derived')}}
  (void)(constder() ? : Base()); // expected-error {{incompatible operand types ('const Derived' and 'Base')}}

  Priv priv;
  Fin fin;
  (void)(Base() ? : Priv()); // expected-error {{private base class}}
  (void)(Priv() ? : Base()); // expected-error {{private base class}}
  (void)(Base() ? : Fin()); // expected-error {{ambiguous conversion from derived class 'Fin' to base class 'Base':}}
  (void)(Fin() ? : Base()); // expected-error {{ambiguous conversion from derived class 'Fin' to base class 'Base':}}
  (void)(base ? : priv); // expected-error {{private base class}}
  (void)(priv ? : base); // expected-error {{private base class}}
  (void)(base ? : fin); // expected-error {{ambiguous conversion from derived class 'Fin' to base class 'Base':}}
  (void)(fin ? : base); // expected-error {{ambiguous conversion from derived class 'Fin' to base class 'Base':}}

  // b2.2 (non-hierarchy)
  b = I() ? : b;
  b = b ? : I();
  I i1(I() ? : J());
  I i2(J() ? : I());
  // "the type [it] would have if E2 were converted to an rvalue"
  vfn pfn = F() ? : p3;
  using Tvfn = decltype(p3 ? : F());
  using Tvfn = vfn;
#if 0
  (void)(A() ? : B()); // expected-error {{conversion from 'B' to 'A' is ambiguous}}
#endif
  (void)(B() ? : A()); // expected-error {{conversion from 'B' to 'A' is ambiguous}}
  (void)(1 ? : Ambig()); // expected-error {{conversion from 'Ambig' to 'int' is ambiguous}}
  (void)(Ambig() ? : 1); // expected-error {{conversion from 'Ambig' to 'int' is ambiguous}}
  // By the way, this isn't an lvalue:
  &(b ? : i1); // expected-error {{cannot take the address of an rvalue}}
}

void p4() {
  // lvalue, same type
  Fields flds;
  int &ir = flds.i1 ? : flds.i2;
  (flds.i1 ? : flds.b1) = 0;
}

void p5() {
  // conversion to built-in types
  double d = I() ? : K();
  vfn pfn = F() ? : G();
  DFnPtr pfm;
  pfm = DFnPtr() ? : &Base::fn1;
  pfm = &Base::fn1 ? : DFnPtr();
}

void p6(int i, int *pi, int &ir) {
  // final conversions
  i = i ? : ir;
  pi = pi ? : 0;
  pi = 0 ? : &i;
  i = i ? : EVal;
  i = EVal ? : i;
  double d = 'c' ? : 4.0;
  using Td = decltype('c' ? : 4.0);
  using Td = decltype(4.0 ? : 'c');
  using Td = double;
  Base *pb = (Base*)0 ? : (Derived*)0;
  pb = (Derived*)0 ? : (Base*)0;
  DFnPtr pfm;
  pfm = &Base::fn1 ? : &Derived::fn2;
  pfm = &Derived::fn2 ? : &Base::fn1;
  pfm = &Derived::fn2 ? : 0;
  pfm = 0 ? : &Derived::fn2;
  const int (MixedFieldsDerived::*mp1) =
    &MixedFields::ci ? : &MixedFieldsDerived::i;
  const volatile int (MixedFields::*mp2) =
    &MixedFields::ci ? : &MixedFields::cvi;
  (void)(&MixedFields::ci ? : &MixedFields::vi);
  // Conversion of primitives does not result in an lvalue.
  &(i ? : d); // expected-error {{cannot take the address of an rvalue}}

  Fields flds;
  (void)&(flds.i1 ? : flds.b1); // expected-error {{address of bit-field requested}}

  unsigned long test0 = 5;
  test0 = (long) test0 ? : test0; // expected-warning {{operand of ? changes signedness: 'long' to 'unsigned long'}}
  test0 = (int) test0 ? : test0; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}
  test0 = (short) test0 ? : test0; // expected-warning {{operand of ? changes signedness: 'short' to 'unsigned long'}}
  test0 = test0 ? : (long) test0; // expected-warning {{operand of ? changes signedness: 'long' to 'unsigned long'}}
  test0 = test0 ? : (int) test0; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}
  test0 = test0 ? : (short) test0; // expected-warning {{operand of ? changes signedness: 'short' to 'unsigned long'}}
  test0 = test0 ? : (long) 10;
  test0 = test0 ? : (int) 10;
  test0 = test0 ? : (short) 10;
  test0 = (long) 10 ? : test0;
  test0 = (int) 10 ? : test0;
  test0 = (short) 10 ? : test0;

  int test1;
  test0 = EVal ? : test0;
  test1 = EVal ? : (int) test0;

  test0 = EVal ? : test1; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}
  test0 = test1 ? : EVal; // expected-warning {{operand of ? changes signedness: 'int' to 'unsigned long'}}

  test1 = EVal ? : (int) test0;
  test1 = (int) test0 ? : EVal;

  // Note the thing that this does not test: since DR446, various situations
  // *must* create a separate temporary copy of class objects. This can only
  // be properly tested at runtime, though.

#if 0
  const Abstract &abstract1 = static_cast<const Abstract&>(Derived1()) ? : Derived2(); // expected-error {{allocating an object of abstract class type 'const Abstract'}}
#endif
  const Abstract &abstract2 = static_cast<const Abstract&>(Derived1()) ? : throw 3;
}

namespace PR6595 {
  struct OtherString {
    OtherString();
    OtherString(const char*);
    explicit operator bool();
  };

  struct String {
    String(const char *);
    String(const OtherString&);
    operator const char*() const;
    explicit operator bool();
  };

  void f(bool Cond, String S, OtherString OS) {
    (void)(S ? : "");
    (void)("" ? : S);
    const char a[1] = {'a'};
    (void)(S ? : a);
    using T = decltype(a ? : S);
    using T = String;
    (void)(OS ? : S);
  }
}

namespace PR6757 {
  struct Foo1 {
    Foo1();
    Foo1(const Foo1&);
  };

  struct Foo2 { };

#if 0
  struct Foo3 {
    Foo3(); // expected-note {{requires 0 arguments}}
    Foo3(Foo3&); // expected-note {{would lose const qualifier}}
  };
#endif

  struct Bar {
    operator const Foo1&() const;
    operator const Foo2&() const;
#if 0
    operator const Foo3&() const;
#endif
    explicit operator bool();
  };

  void f() {
    (void)(Bar() ? : Foo1()); // okay
    (void)(Bar() ? : Foo2()); // okay
#if 0
    (void)(Bar() ? : Foo3()); // expected-error {{no viable constructor copying temporary}}
#endif
  }
}

namespace test1 {
  struct A {
    enum Foo { fa };

    Foo x();
  };

  void foo(int);

  void test(A *a) {
    foo(a->x() ? : 0);
  }
}

namespace rdar7998817 {
  class X {
    X(X&); // expected-note {{declared private here}}

    struct ref { };

  public:
    X();
    X(ref);

    operator ref();
    explicit operator bool();
  };

  void f() {
    X x;
    (void)(x ? : X()); // expected-error {{calling a private constructor of class 'rdar7998817::X'}}
  }
}

namespace PR7598 {
  enum Enum {
    v = 1,
  };

  const Enum g() {
    return v;
  }

  const volatile Enum g2() {
    return v;
  }

  void f() {
    const Enum v2 = v;
    Enum e = g() ? : v;
    Enum e2 = v2 ? : v;
    Enum e3 = g2() ? : v;
  }

}

namespace PR9236 {
#define NULL 0L
  void f() {
    int i;
    (void)(A() ? : NULL); // expected-error {{non-pointer operand type 'A' incompatible with NULL}}
    (void)(NULL ? : A()); // expected-error {{non-pointer operand type 'A' incompatible with NULL}}
    (void)(0 ? : A()); // expected-error {{incompatible operand types}}
    (void)(nullptr ? : A()); // expected-error {{non-pointer operand type 'A' incompatible with nullptr}}
    (void)(nullptr ? : i); // expected-error {{non-pointer operand type 'int' incompatible with nullptr}}
    (void)(__null ? : A()); // expected-error {{non-pointer operand type 'A' incompatible with NULL}}
    (void)((void*)0 ? : A()); // expected-error {{incompatible operand types}}
  }
}

namespace cwg587 {
  template<typename T> void f(T x, const T y) {
    const T *p = &(x ? : y);
  }
  struct S { explicit operator bool(); };
  template void f(int, const int);
  template void f(S, const S);

  void g(int i, const int ci, volatile int vi, const volatile int cvi) {
    const int &cir = i ? : ci;
    volatile int &vir = i ? : vi;
    const volatile int &cvir1 = ci ? : cvi;
    const volatile int &cvir2 = vi ? : cvi;
    const volatile int &cvir3 = ci ? : vi; // expected-error {{volatile lvalue reference to type 'const volatile int' cannot bind to a temporary of type 'int'}}
  }
}

namespace PR17052 {
  struct X {
    int i_;

    int &test() { return i_ ? : throw 1; }
  };
}

namespace PR26448 {
struct Base { explicit operator bool(); } b;
struct Derived : Base {} d;
typedef decltype(static_cast<Base&&>(b) ? : static_cast<Derived&&>(d)) x;
typedef Base &&x;
}

namespace lifetime_extension {
  struct A { explicit operator bool(); };
  struct B : A { B(); ~B(); };
  struct C : A { C(); ~C(); };

  void f() {
    A &&r = static_cast<A&&>(B()) ? : static_cast<A&&>(C());
  }

  struct D {
    A &&a;
    explicit operator bool();
  };
  void f_indirect(bool b) {
    D d = D{B()} ?
                 : D{C()};
    // expected-cxx11-warning@-1 {{temporary whose address is used as value of local variable 'd' will be destroyed at the end of the full-expression}}
  }
}

namespace PR46484 {
void g(int a, int b) {
  long d = a = b ? : throw 0;
}
} // namespace PR46484

namespace GH15998 {
  Enum test(Enum e) {
    return e ? : EVal;
  }
}
