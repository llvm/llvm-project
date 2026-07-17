// RUN: %clang_cc1 -std=c++26 -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1 -std=c++26                                         -verify=both,ref      %s

#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
namespace PaperSample {
  struct Superbase {
    int a = 10;
  };

  struct Common: Superbase {
    unsigned counter = 1337;
  };

  struct Left: virtual Common {
    unsigned value{0};

    constexpr Left() = default;

    constexpr const unsigned & get_counter() const {
      return Common::counter;
    }
  };


  struct Right: virtual Common {
    unsigned value{0};

    constexpr Right() = default;
    constexpr const unsigned & get_counter() const {
      return Common::counter;
    }
  };

  struct Child: Left, Right {
    unsigned x = 12;
    unsigned y = 13;

    constexpr Child() = default;
  };

  constexpr auto ch = Child();
  static_assert(&ch.Left::get_counter() == &ch.Right::get_counter());
  static_assert(ch.counter == 1337);

  static_assert(((Common)ch).counter == 1337);
  static_assert(ch.a == 10);
}

namespace ZeroInit1 {
  struct A {
    int a;
  };

  struct B : public virtual A {
    int b;
  };

  constexpr B b{};
  static_assert(b.b == 0);
  static_assert(b.a == 0);
  static_assert((void*)(A*)&b == (void*)(A*)&b);
}

namespace Destruction {
  struct A {
    int &a;
    constexpr A(int &a) :a(a) {}
    constexpr ~A() { ++a; }
  };

  struct B : public virtual A {
    constexpr B(int &a) : A(a) {}
  };

  constexpr int foo() {
    int m = 0;
    {
      B b(m);
    }
    return m;
  }
  static_assert(foo() == 1);
}


namespace VirtualBaseWithVirtualFunctions {
  struct VBase {
    int x = 5;
    constexpr virtual int compute() const { return x * 2; }
    constexpr virtual ~VBase() = default;
  };

  struct Derived : virtual VBase {
    int y = 3;
    constexpr int compute() const override { return x + y; }
  };

  constexpr bool test_virtual_function() {
    Derived d;
    VBase *ptr = &d;
    return ptr->compute() == 8;
  }

  static_assert(test_virtual_function());
}

namespace DynamicCast {
  struct A {
    virtual constexpr int f() const {return 10;}
  };
  struct B {
    virtual constexpr int f() const {return 20;}
  };
  struct C : virtual A, virtual B {
    constexpr int f() const override { return 30; }
  };

  struct D: C {};
  struct E : D{
    constexpr ~E() {}
  };

  constexpr E e{};
  static_assert(e.f() == 30);

  static_assert((void*)(A*)&e == (void*)(A*)&e);
  static_assert((void*)(A*)&e != (void*)(B*)&e);

  static_assert(dynamic_cast<const B*>(&e) != nullptr);
  static_assert(dynamic_cast<const A*>(&e) != nullptr);

  constexpr const B *b= (B*)&e;
  static_assert(dynamic_cast<const C*>(b) != nullptr);
}

namespace UninitializedFields {

  struct A  {
    int a; // both-note {{declared here}}
    constexpr A() {}
  };
  struct B : public  A {
  };
  constexpr B b{}; // both-error {{must be initialized by a constant expression}} \
                   // both-note {{subobject 'a' is not initialized}}}


  struct X {
    int *p;
    constexpr X() {
       p = new int; // both-note {{heap allocation performed here}}
    }
  };
  struct Y: public virtual X {
  };
  constexpr Y y; // both-error {{must be initialized by a constant expression}} \
                 // both-note {{pointer to heap-allocated object is not a constant expression}}
}


namespace DtorOrder {
  enum {
    R_A = 1,
    R_B = 2,
    R_C = 3,
    R_F = 4,
    R_G = 5,
  };

  struct A {
    int a; int b;
    int *results;
    int &i;

    constexpr A(int *results, int &i) : results(results), i(i) {}
    constexpr ~A() {
      *(results + i) = R_A;
      ++i;
    }

  };
  struct B : public virtual A {
    int c; int d; 
    int *results;
    int &i;

    constexpr B(int *results, int &i) : A(results, i), results(results), i(i) {}
    constexpr ~B() {
      *(results + i) = R_B;
      ++i;
    }
  };


  struct G {
    int *results;
    int &i;
    constexpr G(int *results, int &i) : results(results), i(i) {}

    constexpr ~G() {
      *(results + i) = R_G;
      ++i;
    }
  };


  struct F : virtual G{
    int *results;
    int &i;
    constexpr F(int *results, int &i) : G(results, i), results(results), i(i) {}
    constexpr ~F() {
      *(results + i) = R_F;
      ++i;
    }
  };

  struct C : public virtual A, public virtual B {
    int *results;
    int &i;
    int m = 10;

    F f;

    constexpr C(int *results, int &i) : A(results, i), B(results, i), results(results), i(i), f(results,i) {}

    constexpr ~C() {
      *(results + i) = R_C;
      ++i;
    }
  };

  constexpr int foo() {
    int results[] = {0, 0, 0, 0, 0, 0, 0};

    int i = 0;
    {
     C c = C(results, i);
    }
     return i == 5 &&
            results[0] == R_C &&
            results[1] == R_F &&
            results[2] == R_G &&
            results[3] == R_B &&
            results[4] == R_A;
  }
  static_assert(foo() == 1);
}


namespace CtorOrder {
  enum {
    R_A = 1,
    R_B = 2,
    R_C = 3,
    R_F = 4,
    R_G = 5,
  };

  struct A {
    int a; int b;
    int *results;
    int &i;

    constexpr A(int *results, int &i) : results(results), i(i) {
      *(results + i) = R_A;
      ++i;
    }
  };
  struct B : public virtual A {
    int c; int d; 
    int *results;
    int &i;

    constexpr B(int *results, int &i) : A(results, i), results(results), i(i) {
      *(results + i) = R_B;
      ++i;
    }
  };


  struct G {
    int *results;
    int &i;
    constexpr G(int *results, int &i) : results(results), i(i) {
      *(results + i) = R_G;
      ++i;
    }
  };


  struct F : virtual G{
    int *results;
    int &i;
    constexpr F(int *results, int &i) : G(results, i), results(results), i(i) {
      *(results + i) = R_F;
      ++i;
    }
  };

  struct C : public virtual A, public virtual B {
    int *results;
    int &i;
    int m = 10;

    F f;

    constexpr C(int *results, int &i) : A(results, i), B(results, i), results(results), i(i), f(results,i) {
      *(results + i) = R_C;
      ++i;
    }
  };

  constexpr int foo() {
    int results[] = {0, 0, 0, 0, 0, 0, 0};

    int i = 0;
    {
     C c = C(results, i);
    }
     return i == 5 &&
            results[0] == R_A &&
            results[1] == R_B &&
            results[2] == R_G &&
            results[3] == R_F &&
            results[4] == R_C;
  }
  static_assert(foo() == 1);
}

namespace ImplicitValueInit {
  struct B {int m; };
  struct Ints2 : public virtual B{
    int a = 10;
    int b;
  };
  constexpr Ints2 ints22; // both-error {{without a user-provided default constructor}}
  static_assert(ints22.m == 0);
}

namespace Ctors {

  struct K {
    int k;
    constexpr K(int k) : k(k) {}
  };

  struct A : public virtual K {
    int a;
    constexpr A(int a) : a(a), K(12) {}
  };

  struct B : public virtual A {
    constexpr B() : A(100), K(200) {}
    constexpr B(int) : K(200), A(100) {}
  };

  constexpr B b{};
  static_assert(b.a == 100);
  static_assert(b.k == 200);

  constexpr B b2{-1};
  static_assert(b2.a == 100);
  static_assert(b2.k == 200);

  constexpr A a{13};
  static_assert(a.a == 13);
  static_assert(a.k == 12);
}

namespace Ctors2 {
  struct A {
    constexpr A(int *p, int x) { *p += x; }
  };
  struct B : virtual A {
    constexpr B(int *p) : A(p, 1) {}
  };
  struct C : virtual B {
    constexpr C(int *p) : B(p), A(p, 2) {}
  };
  constexpr int f() {
    int x = 0;
    C c(&x);
    return x;
  }
  static_assert(f() == 2);
}

namespace VirtCalls {
  struct K {
    virtual constexpr int bar() const { return 30; }
  };


  struct X : virtual K {
    virtual constexpr int foo() const { return 10; }
  };

  struct Y : virtual K {};

  struct Z : X, Y {
    constexpr int bar() const override { return 50; }
  };

  constexpr Z z{};
  static_assert(z.foo() == 10);
  static_assert(z.bar() == 50);
}

namespace ConstantDestruction {
  struct V {
    bool b;
    constexpr ~V() {
      __builtin_abort(); // both-note {{subexpression not valid in a constant expression}}
    }
  };

  struct X : virtual V { // expected-note {{in call to}}
    constexpr X() : V(true) {}
  };
  constexpr X x; // both-error {{constexpr variable 'x' must have constant destruction}} \
                 // both-note {{in call to}} \
                 // ref-note {{in call to}}
}

namespace Offsets {
  struct A { int a; };
  struct X { char x[2] = {}; };

  struct B : virtual A {
    int b;
    constexpr B() : A(127), b(123) {}
  };

  struct C : virtual X, B {
    int c;
    constexpr C(int) : c(100), B(), A(128) {}
  };

  constexpr C c{12};
#if !defined(_WIN32)
#if __SIZEOF_SIZE_T__ == 8
  static_assert( (fold((char*)&c.c - (char*)&c)) == 12);
  static_assert( (fold((char*)&c.b - (char*)&c)) == 8);
  static_assert( (fold((char*)&c.a - (char*)&c)) == 20);
  static_assert( (fold((char*)&c.x - (char*)&c)) == 16);
#else
  static_assert( (fold((char*)&c.c - (char*)&c)) == 8);
  static_assert( (fold((char*)&c.b - (char*)&c)) == 4);
  static_assert( (fold((char*)&c.a - (char*)&c)) == 16);
  static_assert( (fold((char*)&c.x - (char*)&c)) == 12);
#endif
# else
  static_assert( (fold((char*)&c.c - (char*)&c)) == 16);
  static_assert( (fold((char*)&c.b - (char*)&c)) == 8);
  static_assert( (fold((char*)&c.a - (char*)&c)) == 28);
  static_assert( (fold((char*)&c.x - (char*)&c)) == 24);
#endif

  static_assert( (fold((char*)&c.c) != fold((char*)&c)));
  static_assert( (fold((char*)&c.a) != fold((char*)&c)));
  static_assert( (fold((char*)&c.a) != fold((char*)&c.c)));
  static_assert( (fold((char*)&c.x) != fold((char*)&c.c)));
  static_assert( (fold((char*)&c.x) != fold((char*)&c)));
  static_assert( (fold((char*)&c.b) != fold((char*)&c)));
  static_assert( (fold((char*)&c.b) != fold((char*)&c.a)));
  static_assert( (fold((char*)&c.b) != fold((char*)&c.x)));
}

namespace LValueAPValue {
  struct V {
    int n = 7;
  };
  struct D : virtual V {};
  constexpr D d{};
  consteval const V *f() { return &d; }
  constexpr int g() { return f()->n; }
  static_assert(g() == 7);
}

namespace MostDerivedIsArrayElem {
  struct V { int n = 7; int k;};
  struct D : virtual V {};
  constexpr D d[9] = {};

  static_assert(d[0].n == 7);
  static_assert(d[0].k == 0);
  static_assert(d[8].n == 7);
  static_assert(d[8].k == 0);
}
