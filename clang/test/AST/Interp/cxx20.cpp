// RUN: %clang_cc1 -fcxx-exceptions -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -std=c++20 -verify=ref %s

void test_alignas_operand() {
  alignas(8) char dummy;
  static_assert(__alignof(dummy) == 8);
}

constexpr int getMinus5() {
  int a = 10;
  a = -5;
  int *p = &a;
  return *p;
}
static_assert(getMinus5() == -5, "");

constexpr int assign() {
  int m = 10;
  int k = 12;

  m = (k = 20);

  return m;
}
static_assert(assign() == 20, "");


constexpr int pointerAssign() {
  int m = 10;
  int *p = &m;

  *p = 12; // modifies m

  return m;
}
static_assert(pointerAssign() == 12, "");

constexpr int pointerDeref() {
  int m = 12;
  int *p = &m;

  return *p;
}
static_assert(pointerDeref() == 12, "");

constexpr int pointerAssign2() {
  int m = 10;
  int *p = &m;
  int **pp = &p;

  **pp = 12;

  int v = **pp;

  return v;
}
static_assert(pointerAssign2() == 12, "");

constexpr int unInitLocal() {
  int a;
  return a; // ref-note {{read of uninitialized object}} \
            // expected-note {{read of uninitialized object}}
}
static_assert(unInitLocal() == 0, ""); // ref-error {{not an integral constant expression}} \
                                       // ref-note {{in call to 'unInitLocal()'}} \
                                       // expected-error {{not an integral constant expression}} \
                                       // expected-note {{in call to 'unInitLocal()'}} \

constexpr int initializedLocal() {
  int a;
  a = 20;
  return a;
}
static_assert(initializedLocal() == 20);

constexpr int initializedLocal2() {
  int a[2];
  return *a; // expected-note {{read of uninitialized object is not allowed in a constant expression}} \
             // ref-note {{read of uninitialized object is not allowed in a constant expression}}
}
static_assert(initializedLocal2() == 20); // expected-error {{not an integral constant expression}} \
                                          // expected-note {{in call to}} \
                                          // ref-error {{not an integral constant expression}} \
                                          // ref-note {{in call to}}


struct Int { int a; };
constexpr int initializedLocal3() {
  Int i;
  return i.a; // ref-note {{read of uninitialized object is not allowed in a constant expression}} \
              // expected-note {{read of uninitialized object}}
}
static_assert(initializedLocal3() == 20); // expected-error {{not an integral constant expression}} \
                                          // expected-note {{in call to}} \
                                          // ref-error {{not an integral constant expression}} \
                                          // ref-note {{in call to}}



#if 0
// FIXME: This code should be rejected because we pass an uninitialized value
//   as a function parameter.
constexpr int inc(int a) { return a + 1; }
constexpr int f() {
    int i;
    return inc(i);
}
static_assert(f());
#endif

/// Distinct literals have disctinct addresses.
/// see https://github.com/llvm/llvm-project/issues/58754
constexpr auto foo(const char *p) { return p; }
constexpr auto p1 = "test1";
constexpr auto p2 = "test2";

constexpr bool b1 = foo(p1) == foo(p1);
static_assert(b1);

constexpr bool b2 = foo(p1) == foo(p2); // ref-error {{must be initialized by a constant expression}} \
                                        // ref-note {{comparison of addresses of literals}} \
                                        // ref-note {{declared here}}
static_assert(!b2); // ref-error {{not an integral constant expression}} \
                    // ref-note {{not a constant expression}}

constexpr auto name1() { return "name1"; }
constexpr auto name2() { return "name2"; }

constexpr auto b3 = name1() == name1();
static_assert(b3);
constexpr auto b4 = name1() == name2(); // ref-error {{must be initialized by a constant expression}} \
                                        // ref-note {{has unspecified value}} \
                                        // ref-note {{declared here}}
static_assert(!b4); // ref-error {{not an integral constant expression}} \
                    // ref-note {{not a constant expression}}

namespace UninitializedFields {
  class A {
  public:
    int a; // expected-note 4{{subobject declared here}} \
           // ref-note 4{{subobject declared here}}
    constexpr A() {}
  };
  constexpr A a; // expected-error {{must be initialized by a constant expression}} \
                 // expected-note {{subobject 'a' is not initialized}} \
                 // ref-error {{must be initialized by a constant expression}} \
                 // ref-note {{subobject 'a' is not initialized}}
  constexpr A aarr[2]; // expected-error {{must be initialized by a constant expression}} \
                       // expected-note {{subobject 'a' is not initialized}} \
                       // ref-error {{must be initialized by a constant expression}} \
                       // ref-note {{subobject 'a' is not initialized}}
  class F {
    public:
      int f; // expected-note 3{{subobject declared here}} \
             // ref-note 3{{subobject declared here}}

      constexpr F() {}
      constexpr F(bool b) {
        if (b)
          f = 42;
      }
  };

  constexpr F foo[2] = {true}; // expected-error {{must be initialized by a constant expression}} \
                               // expected-note {{subobject 'f' is not initialized}} \
                               // ref-error {{must be initialized by a constant expression}} \
                               // ref-note {{subobject 'f' is not initialized}}
  constexpr F foo2[3] = {true, false, true}; // expected-error {{must be initialized by a constant expression}} \
                                             // expected-note {{subobject 'f' is not initialized}} \
                                             // ref-error {{must be initialized by a constant expression}} \
                                             // ref-note {{subobject 'f' is not initialized}}
  constexpr F foo3[3] = {true, true, F()}; // expected-error {{must be initialized by a constant expression}} \
                                           // expected-note {{subobject 'f' is not initialized}} \
                                           // ref-error {{must be initialized by a constant expression}} \
                                           // ref-note {{subobject 'f' is not initialized}}



  class Base {
  public:
    bool b;
    int a; // expected-note {{subobject declared here}} \
           // ref-note {{subobject declared here}}
    constexpr Base() : b(true) {}
  };

  class Derived : public Base {
  public:
    constexpr Derived() : Base() {}   };

  constexpr Derived D; // expected-error {{must be initialized by a constant expression}} \
                       // expected-note {{subobject 'a' is not initialized}} \
                       // ref-error {{must be initialized by a constant expression}} \
                       // ref-note {{subobject 'a' is not initialized}}

  class C2 {
  public:
    A a;
    constexpr C2() {}   };
  constexpr C2 c2; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{subobject 'a' is not initialized}} \
                   // ref-error {{must be initialized by a constant expression}} \
                   // ref-note {{subobject 'a' is not initialized}}

  class C3 {
  public:
    A a[2];
    constexpr C3() {}
  };
  constexpr C3 c3; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{subobject 'a' is not initialized}} \
                   // ref-error {{must be initialized by a constant expression}} \
                   // ref-note {{subobject 'a' is not initialized}}

  class C4 {
  public:
    bool B[2][3]; // expected-note {{subobject declared here}} \
                  // ref-note {{subobject declared here}}
    constexpr C4(){}
  };
  constexpr C4 c4; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{subobject 'B' is not initialized}} \
                   // ref-error {{must be initialized by a constant expression}} \
                   // ref-note {{subobject 'B' is not initialized}}
};

namespace ConstThis {
  class Foo {
    const int T = 12; // expected-note {{declared const here}} \
                      // ref-note {{declared const here}}
    int a;
  public:
    constexpr Foo() { // expected-note {{declared here}}
      this->a = 10;
      T = 13; // expected-error {{cannot assign to non-static data member 'T' with const-qualified type}} \
              // ref-error {{cannot assign to non-static data member 'T' with const-qualified type}}
    }
  };
  constexpr Foo F; // expected-error {{must be initialized by a constant expression}} \
                   // FIXME: The following note is wrong. \
                   // expected-note {{undefined constructor 'Foo' cannot be used in a constant expression}} \
                   // ref-error {{must be initialized by a constant expression}}


  class FooDtor {
    int a;
  public:
    constexpr FooDtor() {
      this->a = 10;
    }
    constexpr ~FooDtor() {
      this->a = 12;
    }
  };

  constexpr int foo() {
    const FooDtor f;
    return 0;
  }
  static_assert(foo() == 0);

  template <bool Good>
  struct ctor_test {
    int a = 0;

    constexpr ctor_test() {
      if (Good)
        a = 10;
      int local = 100 / a; // expected-note {{division by zero}} \
                           // ref-note {{division by zero}}
    }
  };

  template <bool Good>
  struct dtor_test {
    int a = 0;

    constexpr dtor_test() = default;
    constexpr ~dtor_test() {
      if (Good)
        a = 10;
      int local = 100 / a; // expected-note {{division by zero}} \
                           // ref-note {{division by zero}}
    }
  };

  constexpr ctor_test<true> good_ctor;
  constexpr dtor_test<true> good_dtor;

  constexpr ctor_test<false> bad_ctor; // expected-error {{must be initialized by a constant expression}} \
                                       // expected-note {{in call to}} \
                                       // ref-error {{must be initialized by a constant expression}} \
                                       // ref-note {{in call to}}
  constexpr dtor_test<false> bad_dtor; // expected-error {{must have constant destruction}} \
                                       // expected-note {{in call to}} \
                                       // ref-error {{must have constant destruction}} \
                                       // ref-note {{in call to}}
};

namespace BaseInit {
  struct Base {
    int a;
  };

  struct Intermediate : Base {
    int b;
  };

  struct Final : Intermediate {
    int c;

    constexpr Final(int a, int b, int c) : c(c) {}
  };

  static_assert(Final{1, 2, 3}.c == 3, ""); // OK
  static_assert(Final{1, 2, 3}.a == 0, ""); // expected-error {{not an integral constant expression}} \
                                            // expected-note {{read of uninitialized object}} \
                                            // ref-error {{not an integral constant expression}} \
                                            // ref-note {{read of uninitialized object}}


  struct Mixin  {
    int b;

    constexpr Mixin() = default;
    constexpr Mixin(int b) : b(b) {}
  };

  struct Final2 : Base, Mixin {
    int c;

    constexpr Final2(int a, int b, int c) : Mixin(b), c(c) {}
    constexpr Final2(int a, int b, int c, bool) : c(c) {}
  };

  static_assert(Final2{1, 2, 3}.c == 3, ""); // OK
  static_assert(Final2{1, 2, 3}.b == 2, ""); // OK
  static_assert(Final2{1, 2, 3}.a == 0, ""); // expected-error {{not an integral constant expression}} \
                                             // expected-note {{read of uninitialized object}} \
                                             // ref-error {{not an integral constant expression}} \
                                             // ref-note {{read of uninitialized object}}


  struct Mixin3  {
    int b;
  };

  struct Final3 : Base, Mixin3 {
    int c;

    constexpr Final3(int a, int b, int c) : c(c) { this->b = b; }
    constexpr Final3(int a, int b, int c, bool) : c(c) {}
  };

  static_assert(Final3{1, 2, 3}.c == 3, ""); // OK
  static_assert(Final3{1, 2, 3}.b == 2, ""); // OK
  static_assert(Final3{1, 2, 3}.a == 0, ""); // expected-error {{not an integral constant expression}} \
                                             // expected-note {{read of uninitialized object}} \
                                             // ref-error {{not an integral constant expression}} \
                                             // ref-note {{read of uninitialized object}}
};

namespace Destructors {

  class Inc final {
  public:
    int &I;
    constexpr Inc(int &I) : I(I) {}
    constexpr ~Inc() {
      I++;
    }
  };

  class Dec final {
  public:
    int &I;
    constexpr Dec(int &I) : I(I) {}
    constexpr ~Dec() {
      I--;
    }
  };



  constexpr int m() {
    int i = 0;
    {
      Inc f1(i);
      Inc f2(i);
      Inc f3(i);
    }
    return i;
  }
  static_assert(m() == 3, "");


  constexpr int C() {
    int i = 0;

    while (i < 10) {
      Inc inc(i);
      continue;
      Dec dec(i);
    }
    return i;
  }
  static_assert(C() == 10, "");


  constexpr int D() {
    int i = 0;

    {
      Inc i1(i);
      {
        Inc i2(i);
        return i;
      }
    }

    return i;
  }
  static_assert(D() == 0, "");

  constexpr int E() {
    int i = 0;

    for(;;) {
      Inc i1(i);
      break;
    }
    return i;
  }
  static_assert(E() == 1, "");


  /// FIXME: This should be rejected, since we call the destructor
  ///   twice. However, GCC doesn't care either.
  constexpr int ManualDtor() {
    int i = 0;
    {
      Inc I(i); // ref-note {{destroying object 'I' whose lifetime has already ended}}
      I.~Inc();
    }
    return i;
  }
  static_assert(ManualDtor() == 1, ""); // expected-error {{static assertion failed}} \
                                        // expected-note {{evaluates to '2 == 1'}} \
                                        // ref-error {{not an integral constant expression}} \
                                        // ref-note {{in call to 'ManualDtor()'}}

  constexpr void doInc(int &i) {
    Inc I(i);
    return;
  }
  constexpr int testInc() {
    int i = 0;
    doInc(i);
    return i;
  }
  static_assert(testInc() == 1, "");
  constexpr void doInc2(int &i) {
    Inc I(i);
    // No return statement.
  }
   constexpr int testInc2() {
    int i = 0;
    doInc2(i);
    return i;
  }
  static_assert(testInc2() == 1, "");


  namespace DtorOrder {
    class A {
      public:
      int &I;
      constexpr A(int &I) : I(I) {}
      constexpr ~A() {
        I = 1337;
      }
    };

    class B : public A {
      public:
      constexpr B(int &I) : A(I) {}
      constexpr ~B() {
        I = 42;
      }
    };

    constexpr int foo() {
      int i = 0;
      {
        B b(i);
      }
      return i;
    }

    static_assert(foo() == 1337);
  }

  class FieldDtor1 {
  public:
    Inc I1;
    Inc I2;
    constexpr FieldDtor1(int &I) : I1(I), I2(I){}
  };

  constexpr int foo2() {
    int i = 0;
    {
      FieldDtor1 FD1(i);
    }
    return i;
  }

  static_assert(foo2() == 2);

  class FieldDtor2 {
  public:
    Inc Incs[3];
    constexpr FieldDtor2(int &I)  : Incs{Inc(I), Inc(I), Inc(I)} {}
  };

  constexpr int foo3() {
    int i = 0;
    {
      FieldDtor2 FD2(i);
    }
    return i;
  }

  static_assert(foo3() == 3);

  struct ArrD {
    int index;
    int *arr;
    int &p;
    constexpr ~ArrD() {
      arr[p] = index;
      ++p;
    }
  };
  constexpr bool ArrayOrder() {
    int order[3] = {0, 0, 0};
    int p = 0;
    {
      ArrD ds[3] = {
        {1, order, p},
        {2, order, p},
        {3, order, p},
      };
      // ds will be destroyed.
    }
    return order[0] == 3 && order[1] == 2 && order[2] == 1;
  }
  static_assert(ArrayOrder());


  // Static members aren't destroyed.
  class Dec2 {
  public:
    int A = 0;
    constexpr ~Dec2() {
      A++;
    }
  };
  class Foo {
  public:
    static constexpr Dec2 a;
    static Dec2 b;
  };
  static_assert(Foo::a.A == 0);
  constexpr bool f() {
    Foo f;
    return true;
  }
  static_assert(Foo::a.A == 0);
  static_assert(f());
  static_assert(Foo::a.A == 0);


  struct NotConstexpr {
    NotConstexpr() {}
    ~NotConstexpr() {}
  };

  struct Outer {
    constexpr Outer() = default;
    constexpr ~Outer();

    constexpr int foo() {
      return 12;
    }

    constexpr int bar()const  {
      return Outer{}.foo();
    }

    static NotConstexpr Val;
  };

  constexpr Outer::~Outer() {}

  constexpr Outer O;
  static_assert(O.bar() == 12);
}

namespace BaseAndFieldInit {
  struct A {
    int a;
  };

  struct B : A {
    int b;
  };

  struct C : B {
    int c;
  };

  constexpr C c = {1,2,3};
  static_assert(c.a == 1 && c.b == 2 && c.c == 3);
}

namespace ImplicitFunction {
  struct A {
    int a; // ref-note {{subobject declared here}}
  };

  constexpr int callMe() {
   A a;
   A b{12};

   /// The operator= call here will fail and the diagnostics should be fine.
   b = a; // ref-note {{subobject 'a' is not initialized}} \
          // ref-note {{in call to}} \
          // expected-note {{read of uninitialized object}} \
          // expected-note {{in call to}}

   return 1;
  }
  static_assert(callMe() == 1, ""); // ref-error {{not an integral constant expression}} \
                                    // ref-note {{in call to 'callMe()'}} \
                                    // expected-error {{not an integral constant expression}} \
                                    // expected-note {{in call to 'callMe()'}}
}

/// FIXME: Unfortunately, the similar tests in test/SemaCXX/{compare-cxx2a.cpp use member pointers,
/// which we don't support yet.
namespace std {
  class strong_ordering {
  public:
    int n;
    static const strong_ordering less, equal, greater;
    constexpr bool operator==(int n) const noexcept { return this->n == n;}
    constexpr bool operator!=(int n) const noexcept { return this->n != n;}
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};

  class partial_ordering {
  public:
    long n;
    static const partial_ordering less, equal, greater, equivalent, unordered;
    constexpr bool operator==(long n) const noexcept { return this->n == n;}
    constexpr bool operator!=(long n) const noexcept { return this->n != n;}
  };
  constexpr partial_ordering partial_ordering::less = {-1};
  constexpr partial_ordering partial_ordering::equal = {0};
  constexpr partial_ordering partial_ordering::greater = {1};
  constexpr partial_ordering partial_ordering::equivalent = {0};
  constexpr partial_ordering partial_ordering::unordered = {-127};
} // namespace std

namespace ThreeWayCmp {
  static_assert(1 <=> 2 == -1, "");
  static_assert(1 <=> 1 == 0, "");
  static_assert(2 <=> 1 == 1, "");
  static_assert(1.0 <=> 2.f == -1, "");
  static_assert(1.0 <=> 1.0 == 0, "");
  static_assert(2.0 <=> 1.0 == 1, "");
  constexpr int k = (1 <=> 1, 0); // expected-warning {{comparison result unused}} \
                                  // ref-warning {{comparison result unused}}
  static_assert(k== 0, "");

  /// Pointers.
  constexpr int a[] = {1,2,3};
  constexpr int b[] = {1,2,3};
  constexpr const int *pa1 = &a[1];
  constexpr const int *pa2 = &a[2];
  constexpr const int *pb1 = &b[1];
  static_assert(pa1 <=> pb1 != 0, ""); // expected-error {{not an integral constant expression}} \
                                       // expected-note {{has unspecified value}} \
                                       // ref-error {{not an integral constant expression}} \
                                       // ref-note {{has unspecified value}}
  static_assert(pa1 <=> pa1 == 0, "");
  static_assert(pa1 <=> pa2 == -1, "");
  static_assert(pa2 <=> pa1 == 1, "");
}

// FIXME: Interp should also be able to evaluate this snippet properly.
namespace ConstexprArrayInitLoopExprDestructors
{
  struct Highlander {
      int *p = 0;
      constexpr Highlander() {}
      constexpr void set(int *p) { this->p = p; ++*p; if (*p != 1) throw "there can be only one"; } // expected-note {{not valid in a constant expression}}
      constexpr ~Highlander() { --*p; }
  };

  struct X {
      int *p;
      constexpr X(int *p) : p(p) {}
      constexpr X(const X &x, Highlander &&h = Highlander()) : p(x.p) {
          h.set(p); // expected-note {{in call to '&Highlander()->set(&n)'}}
      }
  };

  constexpr int f() {
      int n = 0;
      X x[3] = {&n, &n, &n};
      auto [a, b, c] = x; // expected-note {{in call to 'X(x[0], Highlander())'}}
      return n;
  }

  static_assert(f() == 0); // expected-error {{not an integral constant expression}} \
                           // expected-note {{in call to 'f()'}}

  int main() {
      return f();
  }
}
