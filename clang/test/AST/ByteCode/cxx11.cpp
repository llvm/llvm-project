// RUN: %clang_cc1 -triple x86_64-linux -verify=both,expected -std=c++11 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux -verify=both,ref      -std=c++11 %s

namespace IntOrEnum {
  const int k = 0;
  const int &p = k;
  template<int n> struct S {};
  S<p> s;
}

const int cval = 2;
template <int> struct C{};
template struct C<cval>;


/// FIXME: This example does not get properly diagnosed in the new interpreter.
extern const int recurse1;
const int recurse2 = recurse1; // both-note {{here}}
const int recurse1 = 1;
int array1[recurse1];
int array2[recurse2]; // both-warning {{variable length arrays in C++}} \
                      // both-note {{initializer of 'recurse2' is not a constant expression}} \
                      // expected-error {{variable length array declaration not allowed at file scope}} \
                      // ref-warning {{variable length array folded to constant array as an extension}}

constexpr int b = b; // both-error {{must be initialized by a constant expression}} \
                     // both-note {{read of object outside its lifetime is not allowed in a constant expression}}


[[clang::require_constant_initialization]] int c = c; // both-error {{variable does not have a constant initializer}} \
                                                      // both-note {{attribute here}} \
                                                      // both-note {{read of non-const variable}} \
                                                      // both-note {{declared here}}


struct S {
  int m;
};
constexpr S s = { 5 };
constexpr const int *p = &s.m + 1;

constexpr const int *np2 = &(*(int(*)[4])nullptr)[0]; // both-error {{constexpr variable 'np2' must be initialized by a constant expression}} \
                                                      // both-note  {{dereferencing a null pointer is not allowed in a constant expression}}

constexpr int preDec(int x) { // both-error {{never produces a constant expression}}
  return --x;                 // both-note {{subexpression}}
}

constexpr int postDec(int x) { // both-error {{never produces a constant expression}}
  return x--;                  // both-note {{subexpression}}
}

constexpr int preInc(int x) { // both-error {{never produces a constant expression}}
  return ++x;                  // both-note {{subexpression}}
}

constexpr int postInc(int x) { // both-error {{never produces a constant expression}}
  return x++;                  // both-note {{subexpression}}
}


namespace ReferenceToConst {
  template<int n> struct S; // both-note 1{{here}}
  struct LiteralType {
    constexpr LiteralType(int n) : n(n) {}
    int n;
  };
  template<int n> struct T {
    T() {
      static const int ki = 42;
      const int &i2 = ki;
      typename S<i2>::T check5; // both-error {{undefined template}}
    }
  };
}



namespace GH50055 {
// Enums without fixed underlying type
enum E1 {e11=-4, e12=4};
enum E2 {e21=0, e22=4};
enum E3 {e31=-4, e32=1024};
enum E4 {e41=0};
// Empty but as-if it had a single enumerator with value 0
enum EEmpty {};

// Enum with fixed underlying type because the underlying type is explicitly specified
enum EFixed : int {efixed1=-4, efixed2=4};
// Enum with fixed underlying type because it is scoped
enum class EScoped {escoped1=-4, escoped2=4};

enum EMaxInt {emaxint1=-1, emaxint2=__INT_MAX__};

enum NumberType {};

E2 testDefaultArgForParam(E2 e2Param = (E2)-1) { // ok, not a constant expression context
  E2 e2LocalInit = e2Param; // ok, not a constant expression context
  return e2LocalInit;
}

// #include <enum-constexpr-conversion-system-header.h>

void testValueInRangeOfEnumerationValues() {
  constexpr E1 x1 = static_cast<E1>(-8);
  constexpr E1 x2 = static_cast<E1>(8);
  // both-error@-1 {{constexpr variable 'x2' must be initialized by a constant expression}}
  // both-note@-2 {{integer value 8 is outside the valid range of values [-8, 7] for the enumeration type 'E1'}}
  E1 x2b = static_cast<E1>(8); // ok, not a constant expression context

  constexpr E2 x3 = static_cast<E2>(-8);
  // both-error@-1 {{constexpr variable 'x3' must be initialized by a constant expression}}
  // both-note@-2 {{integer value -8 is outside the valid range of values [0, 7] for the enumeration type 'E2'}}
  constexpr E2 x4 = static_cast<E2>(0);
  constexpr E2 x5 = static_cast<E2>(8);
  // both-error@-1 {{constexpr variable 'x5' must be initialized by a constant expression}}
  // both-note@-2 {{integer value 8 is outside the valid range of values [0, 7] for the enumeration type 'E2'}}

  constexpr E3 x6 = static_cast<E3>(-2048);
  constexpr E3 x7 = static_cast<E3>(-8);
  constexpr E3 x8 = static_cast<E3>(0);
  constexpr E3 x9 = static_cast<E3>(8);
  constexpr E3 x10 = static_cast<E3>(2048);
  // both-error@-1 {{constexpr variable 'x10' must be initialized by a constant expression}}
  // both-note@-2 {{integer value 2048 is outside the valid range of values [-2048, 2047] for the enumeration type 'E3'}}

  constexpr E4 x11 = static_cast<E4>(0);
  constexpr E4 x12 = static_cast<E4>(1);
  constexpr E4 x13 = static_cast<E4>(2);
  // both-error@-1 {{constexpr variable 'x13' must be initialized by a constant expression}}
  // both-note@-2 {{integer value 2 is outside the valid range of values [0, 1] for the enumeration type 'E4'}}

  constexpr EEmpty x14 = static_cast<EEmpty>(0);
  constexpr EEmpty x15 = static_cast<EEmpty>(1);
  constexpr EEmpty x16 = static_cast<EEmpty>(2);
  // both-error@-1 {{constexpr variable 'x16' must be initialized by a constant expression}}
  // both-note@-2 {{integer value 2 is outside the valid range of values [0, 1] for the enumeration type 'EEmpty'}}

  constexpr EFixed x17 = static_cast<EFixed>(100);
  constexpr EScoped x18 = static_cast<EScoped>(100);

  constexpr EMaxInt x19 = static_cast<EMaxInt>(__INT_MAX__-1);
  constexpr EMaxInt x20 = static_cast<EMaxInt>((long)__INT_MAX__+1);
  // both-error@-1 {{constexpr variable 'x20' must be initialized by a constant expression}}
  // both-note@-2 {{integer value 2147483648 is outside the valid range of values [-2147483648, 2147483647] for the enumeration type 'EMaxInt'}}

  const NumberType neg_one = (NumberType) ((NumberType) 0 - (NumberType) 1); // ok, not a constant expression context
}

template<class T, unsigned size> struct Bitfield {
  static constexpr T max = static_cast<T>((1 << size) - 1);
  // both-error@-1 {{constexpr variable 'max' must be initialized by a constant expression}}
  // both-note@-2 {{integer value 15 is outside the valid range of values [0, 7] for the enumeration type 'E2'}}
};

void testValueInRangeOfEnumerationValuesViaTemplate() {
  Bitfield<E2, 3> good;
  Bitfield<E2, 4> bad; // both-note {{in instantiation}}
}

enum SortOrder {
  AscendingOrder,
  DescendingOrder
};

class A {
  static void f(SortOrder order);
};

void A::f(SortOrder order) {
  if (order == SortOrder(-1)) // ok, not a constant expression context
    return;
}
}

namespace FinalLtorDiags {
  template<int*> struct A {}; // both-note {{template parameter is declared here}}
  int k;
  int *q = &k; // both-note {{declared here}}
  A<q> c; // both-error {{non-type template argument of type 'int *' is not a constant expression}} \
          // both-note {{read of non-constexpr variable 'q' is not allowed in a constant expression}}
}

void lambdas() {
  int d;
  int a9[1] = {[d = 0] = 1}; // both-error {{not an integral constant expression}}
}


namespace InitLinkToRVO {
  struct A {
    int y = 3;
    int z = 1 + y;
  };

  constexpr A make() { return A {}; }
  static_assert(make().z == 4, "");
}

namespace DynamicCast {
  struct S { int x, y; } s;
  constexpr S* sptr = &s;
  struct Str {
    int b : reinterpret_cast<S*>(sptr) == reinterpret_cast<S*>(sptr);
    int g : (S*)(void*)(sptr) == sptr;
  };
}

namespace GlobalInitializer {
  extern int &g; // both-note {{here}}
  struct S {
    int G : g; // both-error {{constant expression}} \
               // both-note {{initializer of 'g' is unknown}}
  };
}

namespace ExternPointer {
  struct S { int a; };
  extern const S pu;
  constexpr const int *pua = &pu.a; // Ok.
}

namespace PseudoDtor {
  typedef int I;
  constexpr int f(int a = 1) { // both-error {{never produces a constant expression}} \
                               // ref-note {{destroying object 'a' whose lifetime has already ended}}
    return (
        a.~I(), // both-note {{pseudo-destructor call is not permitted}} \
                // expected-note {{pseudo-destructor call is not permitted}}
        0);
  }
  static_assert(f() == 0, ""); // both-error {{constant expression}} \
                               // expected-note {{in call to}}
}

namespace IntToPtrCast {
  typedef __INTPTR_TYPE__ intptr_t;

  constexpr intptr_t f(intptr_t x) {
    return (((x) >> 21) * 8);
  }

  extern "C" int foo;
  constexpr intptr_t i = f((intptr_t)&foo - 10); // both-error{{constexpr variable 'i' must be initialized by a constant expression}} \
                                                 // both-note{{reinterpret_cast}}
}

namespace Volatile {
  constexpr int f(volatile int &&r) {
    return r; // both-note {{read of volatile-qualified type 'volatile int'}}
  }
  struct S {
    int j : f(0); // both-error {{constant expression}} \
                  // both-note {{in call to 'f(0)'}}
  };
}

namespace ZeroSizeCmp {
  extern void (*start[])();
  extern void (*end[])();
  static_assert(&start != &end, ""); // both-error {{constant expression}} \
                                     // both-note {{comparison of pointers '&start' and '&end' to unrelated zero-sized objects}}
}

namespace OverlappingStrings {
  static_assert(+"foo" != +"bar", "");
  static_assert(&"xfoo"[1] != &"yfoo"[1], "");
  static_assert(+"foot" != +"foo", "");
  static_assert(+"foo\0bar" != +"foo\0baz", "");


#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
  static_assert(fold((const char*)u"A" != (const char*)"\0A\0x"), "");
  static_assert(fold((const char*)u"A" != (const char*)"A\0\0x"), "");
  static_assert(fold((const char*)u"AAA" != (const char*)"AAA\0\0x"), "");

  constexpr const char *string = "hello";
  constexpr const char *also_string = string;
  static_assert(string == string, "");
  static_assert(string == also_string, "");


  // These strings may overlap, and so the result of the comparison is unknown.
  constexpr bool may_overlap_1 = +"foo" == +"foo"; // both-error {{}} both-note {{addresses of potentially overlapping literals}}
  constexpr bool may_overlap_2 = +"foo" == +"foo\0bar"; // both-error {{}} both-note {{addresses of potentially overlapping literals}}
  constexpr bool may_overlap_3 = +"foo" == &"bar\0foo"[4]; // both-error {{}} both-note {{addresses of potentially overlapping literals}}
  constexpr bool may_overlap_4 = &"xfoo"[1] == &"xfoo"[1]; // both-error {{}} both-note {{addresses of potentially overlapping literals}}


  /// Used to crash.
  const bool x = &"ab"[0] == &"ba"[3];

}

namespace NonConstLocal {
  int a() {
    const int t=t; // both-note {{declared here}}

    switch(1) {
      case t:; // both-note {{initializer of 't' is not a constant expression}} \
               // both-error {{case value is not a constant expression}}
    }
  }
}

#define ATTR __attribute__((require_constant_initialization))
int somefunc() {
  const int non_global = 42; // both-note {{declared here}}
  ATTR static const int &local_init = non_global; // both-error {{variable does not have a constant initializer}} \
                                                  // both-note {{required by}} \
                                                  // both-note {{reference to 'non_global' is not a constant expression}}
}

namespace PR19010 {
  struct Empty {};
  struct Empty2 : Empty {};
  struct Test : Empty2 {
    constexpr Test() {}
    Empty2 array[2];
  };
  void test() { constexpr Test t; }
}

namespace ReadMutableInCopyCtor {
  struct G {
    struct X {};
    union U { X a; };
    mutable U u; // both-note {{declared here}}
  };
  constexpr G g1 = {};
  constexpr G g2 = g1; // both-error {{must be initialized by a constant expression}} \
                       // both-note {{read of mutable member 'u'}} \
                       // both-note {{in call to 'G(g1)'}}
}

namespace GH150709 {
  struct C { };
  struct D : C {
    constexpr int f() const { return 1; };
  };
  struct E : C { };
  struct F : D { };
  struct G : E { };
  
  constexpr C c1, c2[2];
  constexpr D d1, d2[2];
  constexpr E e1, e2[2];
  constexpr F f;
  constexpr G g;

  constexpr auto mp = static_cast<int (C::*)() const>(&D::f);

  // sanity checks for fix of GH150709 (unchanged behavior)
  static_assert((c1.*mp)() == 1, ""); // both-error {{constant expression}}
  static_assert((d1.*mp)() == 1, "");
  static_assert((f.*mp)() == 1, "");
  static_assert((c2[0].*mp)() == 1, ""); // ref-error {{constant expression}}
  static_assert((d2[0].*mp)() == 1, "");

  // incorrectly undiagnosed before fix of GH150709
  static_assert((e1.*mp)() == 1, ""); // ref-error {{constant expression}}
  static_assert((e2[0].*mp)() == 1, ""); // ref-error {{constant expression}}
  static_assert((g.*mp)() == 1, ""); // ref-error {{constant expression}}
}
