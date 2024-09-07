// RUN: %clang_cc1 -triple x86_64-linux -fexperimental-new-constant-interpreter -verify=both,expected -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-linux -verify=both,ref -std=c++11 %s

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

struct S {
  int m;
};
constexpr S s = { 5 };
constexpr const int *p = &s.m + 1;

constexpr const int *np2 = &(*(int(*)[4])nullptr)[0]; // ok

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
