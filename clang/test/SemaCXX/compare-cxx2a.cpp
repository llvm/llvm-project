// Force x86-64 because some of our heuristics are actually based
// on integer sizes.

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fcxx-exceptions -fsyntax-only -pedantic -verify -Wsign-compare -std=c++2a %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fcxx-exceptions -std=c++2a -x c++ %S/Inputs/std-compare.h -emit-pch -o %t.pch
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fcxx-exceptions -fsyntax-only -pedantic -verify -Wsign-compare -std=c++2a %s -include-pch %t.pch

#include "Inputs/std-compare.h"

#define ASSERT_TYPE(...) static_assert(__is_same(__VA_ARGS__))
#define ASSERT_EXPR_TYPE(Expr, Expect) static_assert(__is_same(decltype(Expr), Expect));

struct S {
  static int x[5];
};

void self_compare() {
  int a;
  int *b = nullptr;
  S s;

  (void)(a <=> a); // expected-warning {{self-comparison always evaluates to 'std::strong_ordering::equal'}}
  (void)(b <=> b); // expected-warning {{self-comparison always evaluates to 'std::strong_ordering::equal'}}
  (void)(s.x[a] <=> S::x[a]); // expected-warning {{self-comparison always evaluates to 'std::strong_ordering::equal'}}
}

void test0(long a, unsigned long b) {
  enum EnumA : int {A};
  enum EnumB {B};
  enum EnumC {C = 0x10000};

  (void)((short)a <=> (unsigned short)b);

  // (a,b)
  (void)(a <=> (unsigned long)b); // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)(a <=> (unsigned int) b);
  (void)(a <=> (unsigned short) b);
  (void)(a <=> (unsigned char) b);
  (void)((long)a <=> b);                // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((int)a <=> b);                 // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((short)a <=> b);               // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((signed char)a <=> b);         // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((long)a <=> (unsigned long)b); // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((int)a <=> (unsigned int)b);   // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((short) a <=> (unsigned short) b);
  (void)((signed char) a <=> (unsigned char) b);

  // (A,b)
  (void)(A <=> (unsigned long) b);
  (void)(A <=> (unsigned int) b);
  (void)(A <=> (unsigned short) b);
  (void)(A <=> (unsigned char) b);
  (void)((long) A <=> b);
  (void)((int) A <=> b);
  (void)((short) A <=> b);
  (void)((signed char) A <=> b);
  (void)((long) A <=> (unsigned long) b);
  (void)((int) A <=> (unsigned int) b);
  (void)((short) A <=> (unsigned short) b);
  (void)((signed char) A <=> (unsigned char) b);

  // (a,B)
  (void)(a <=> (unsigned long) B); // expected-error {{argument to 'operator<=>' cannot be narrowed from type 'long' to 'unsigned long'}}
  (void)(a <=> (unsigned int) B);
  (void)(a <=> (unsigned short) B);
  (void)(a <=> (unsigned char) B);
  (void)((long) a <=> B);
  (void)((int) a <=> B);
  (void)((short) a <=> B);
  (void)((signed char) a <=> B);
  (void)((long) a <=> (unsigned long) B); // expected-error {{argument to 'operator<=>' cannot be narrowed from type 'long' to 'unsigned long'}}
  (void)((int) a <=> (unsigned int) B); // expected-error {{argument to 'operator<=>' cannot be narrowed from type 'int' to 'unsigned int'}}
  (void)((short) a <=> (unsigned short) B);
  (void)((signed char) a <=> (unsigned char) B);

  // (C,b)
  (void)(C <=> (unsigned long) b);
  (void)(C <=> (unsigned int) b);
  (void)(C <=> (unsigned short) b); // expected-warning {{comparison of constant 'C' (65536) with expression of type 'unsigned short' is always 'std::strong_ordering::greater'}}
  (void)(C <=> (unsigned char) b);  // expected-warning {{comparison of constant 'C' (65536) with expression of type 'unsigned char' is always 'std::strong_ordering::greater'}}
  (void)((long) C <=> b);
  (void)((int) C <=> b);
  (void)((short) C <=> b);
  (void)((signed char) C <=> b);
  (void)((long) C <=> (unsigned long) b);
  (void)((int) C <=> (unsigned int) b);
  (void)((short) C <=> (unsigned short) b);
  (void)((signed char) C <=> (unsigned char) b);

  // (a,C)
  (void)(a <=> (unsigned long) C); // expected-error {{argument to 'operator<=>' cannot be narrowed from type 'long' to 'unsigned long'}}
  (void)(a <=> (unsigned int) C);
  (void)(a <=> (unsigned short) C);
  (void)(a <=> (unsigned char) C);
  (void)((long) a <=> C);
  (void)((int) a <=> C);
  (void)((short) a <=> C); // expected-warning {{comparison of constant 'C' (65536) with expression of type 'short' is always 'std::strong_ordering::less'}}
  (void)((signed char) a <=> C); // expected-warning {{comparison of constant 'C' (65536) with expression of type 'signed char' is always 'std::strong_ordering::less'}}
  (void)((long) a <=> (unsigned long) C); // expected-error {{argument to 'operator<=>' cannot be narrowed from type 'long' to 'unsigned long'}}
  (void)((int) a <=> (unsigned int) C); // expected-error {{argument to 'operator<=>' cannot be narrowed from type 'int' to 'unsigned int'}}
  (void)((short) a <=> (unsigned short) C);
  (void)((signed char) a <=> (unsigned char) C);

  // (0x80000,b)
  (void)(0x80000 <=> (unsigned long) b);
  (void)(0x80000 <=> (unsigned int) b);
  (void)(0x80000 <=> (unsigned short) b); // expected-warning {{result of comparison of constant 524288 with expression of type 'unsigned short' is always 'std::strong_ordering::greater'}}
  (void)(0x80000 <=> (unsigned char) b); // expected-warning {{result of comparison of constant 524288 with expression of type 'unsigned char' is always 'std::strong_ordering::greater'}}
  (void)((long) 0x80000 <=> b);
  (void)((int) 0x80000 <=> b);
  (void)((short) 0x80000 <=> b);
  (void)((signed char) 0x80000 <=> b);
  (void)((long) 0x80000 <=> (unsigned long) b);
  (void)((int) 0x80000 <=> (unsigned int) b);
  (void)((short) 0x80000 <=> (unsigned short) b);
  (void)((signed char) 0x80000 <=> (unsigned char) b);

  // (a,0x80000)
  (void)(a <=> (unsigned long)0x80000); // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)(a <=> (unsigned int) 0x80000);
  (void)(a <=> (unsigned short) 0x80000);
  (void)(a <=> (unsigned char) 0x80000);
  (void)((long) a <=> 0x80000);
  (void)((int) a <=> 0x80000);
  (void)((short) a <=> 0x80000); // expected-warning {{comparison of constant 524288 with expression of type 'short' is always 'std::strong_ordering::less'}}
  (void)((signed char) a <=> 0x80000); // expected-warning {{comparison of constant 524288 with expression of type 'signed char' is always 'std::strong_ordering::less'}}
  (void)((long)a <=> (unsigned long)0x80000); // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((int)a <=> (unsigned int)0x80000);   // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((short) a <=> (unsigned short) 0x80000);
  (void)((signed char) a <=> (unsigned char) 0x80000);
}

void test5(bool b, bool b2) {
  enum EnumA { A };
  (void)(b <=> b2);      // OK
  (void)(true <=> b);    // OK
  (void)(b <=> -10);     // expected-error {{invalid operands to binary expression ('bool' and 'int')}}
  (void)(b <=> char(1)); // expected-error {{invalid operands to binary expression ('bool' and 'char')}}
  (void)(b <=> A);       // expected-error {{invalid operands to binary expression ('bool' and 'EnumA')}}

  // FIXME: Should this be accepted when narrowing doesn't occur?
  (void)(b <=> 0); // expected-error {{invalid operands to binary expression ('bool' and 'int')}}
  (void)(b <=> 1); // expected-error {{invalid operands to binary expression ('bool' and 'int')}}
}

void test6(signed char sc) {
  (void)(sc <=> 200); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always 'std::strong_ordering::less'}}
  (void)(200 <=> sc); // expected-warning{{comparison of constant 200 with expression of type 'signed char' is always 'std::strong_ordering::greater'}}
}

// Test many signedness combinations.
void test7(unsigned long other) {
  // Common unsigned, other unsigned, constant unsigned
  (void)((unsigned)other <=> (unsigned long)(0x1'ffff'ffff)); // expected-warning{{less}}
  (void)((unsigned)other <=> (unsigned long)(0xffff'ffff));
  (void)((unsigned long)other <=> (unsigned)(0x1'ffff'ffff));
  (void)((unsigned long)other <=> (unsigned)(0xffff'ffff));

  // Common unsigned, other signed, constant unsigned
  (void)((int)other <=> (unsigned long)(0xffff'ffff'ffff'ffff)); // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((int)other <=> (unsigned long)(0x0000'0000'ffff'ffff)); // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((int)other <=> (unsigned long)(0x0000'0000'0fff'ffff)); // expected-error {{argument to 'operator<=>' cannot be narrowed}}
  (void)((int)other <=> (unsigned)(0x8000'0000));                // expected-error {{argument to 'operator<=>' cannot be narrowed}}

  // Common unsigned, other unsigned, constant signed
  (void)((unsigned long)other <=> (int)(0xffff'ffff)); // expected-error {{argument to 'operator<=>' evaluates to -1, which cannot be narrowed to type 'unsigned long'}}

  // Common unsigned, other signed, constant signed
  // Should not be possible as the common type should also be signed.

  // Common signed, other signed, constant signed
  (void)((int)other <=> (long)(0xffff'ffff));           // expected-warning{{less}}
  (void)((int)other <=> (long)(0xffff'ffff'0000'0000)); // expected-warning{{greater}}
  (void)((int)other <=> (long)(0x0fff'ffff));
  (void)((int)other <=> (long)(0xffff'ffff'f000'0000));

  // Common signed, other signed, constant unsigned
  (void)((int)other <=> (unsigned char)(0xffff));
  (void)((int)other <=> (unsigned char)(0xff));

  // Common signed, other unsigned, constant signed
  (void)((unsigned char)other <=> (int)(0xff));
  (void)((unsigned char)other <=> (int)(0xffff));  // expected-warning{{less}}

  // Common signed, other unsigned, constant unsigned
  (void)((unsigned char)other <=> (unsigned short)(0xff));
  (void)((unsigned char)other <=> (unsigned short)(0x100)); // expected-warning{{less}}
  (void)((unsigned short)other <=> (unsigned char)(0xff));
}

void test8(void *vp, const void *cvp, int *ip) {
  (void)(vp <=> cvp); // OK, void* comparisons are allowed.
  (void)(vp <=> ip);
  (void)(ip <=> cvp);
}

void test9(long double ld, double d, float f, int i, long long ll) {
  (void)(f <=> ll); // OK, floating-point to integer is OK
  (void)(d <=> ld);
  (void)(i <=> f);
}

typedef int *INTPTR;
void test_typedef_bug(int *x, INTPTR y) {
  (void)(x <=> y);
}

using nullptr_t = decltype(nullptr);

struct Class {};
struct ClassB : Class {};
struct Class2 {};
using FnTy = void(int);
using MemFnTy = void (Class::*)() const;
using MemDataTy = long(Class::*);

void test_nullptr(int *x, FnTy *fp, MemFnTy memp, MemDataTy memdp) {
  auto r1 = (nullptr <=> nullptr); // expected-error {{invalid operands}}
  auto r2 = (nullptr <=> x); // expected-error {{invalid operands}}
  auto r3 = (fp <=> nullptr); // expected-error {{invalid operands}}
  auto r4 = (0 <=> fp); // expected-error {{ordered comparison between pointer and zero}}
  auto r5 = (nullptr <=> memp); // expected-error {{invalid operands}}
  auto r6 = (0 <=> memdp); // expected-error {{invalid operands}}
  auto r7 = (0 <=> nullptr); // expected-error {{invalid operands}}
}

void test_memptr(MemFnTy mf, MemDataTy md) {
  (void)(mf <=> mf); // expected-error {{invalid operands}} expected-warning {{self-comparison}}
  (void)(md <=> md); // expected-error {{invalid operands}} expected-warning {{self-comparison}}
}

// Test that variable narrowing is deferred for value dependent expressions
template <int Val>
auto test_template_overflow() {
  // expected-error@+1 {{argument to 'operator<=>' evaluates to -1, which cannot be narrowed to type 'unsigned long'}}
  return (Val <=> (unsigned long)0);
}
template auto test_template_overflow<0>();
template auto test_template_overflow<-1>(); // expected-note {{requested here}}

void test_enum_integral_compare() {
  enum EnumA : int {A, ANeg = -1, AMax = __INT_MAX__};
  enum EnumB : unsigned {B, BMax = __UINT32_MAX__ };
  enum EnumC : int {C = -1, C0 = 0};

  (void)(A <=> C); // expected-error {{invalid operands to binary expression ('EnumA' and 'EnumC')}}

  (void)(A <=> (unsigned)0);
  (void)((unsigned)0 <=> A);
  (void)(ANeg <=> (unsigned)0); // expected-error {{argument to 'operator<=>' evaluates to -1, which cannot be narrowed to type 'unsigned int'}}
  (void)((unsigned)0 <=> ANeg); // expected-error {{cannot be narrowed}}

  (void)(B <=> 42);
  (void)(42 <=> B);
  (void)(B <=> (unsigned long long)42);
  (void)(B <=> -1); // expected-error {{argument to 'operator<=>' evaluates to -1, which cannot be narrowed to type 'unsigned int'}}
  (void)(BMax <=> (unsigned long)-1);

  (void)(C0 <=> (unsigned)42);
  (void)(C <=> (unsigned)42); // expected-error {{argument to 'operator<=>' evaluates to -1, which cannot be narrowed to type 'unsigned int'}}
}

namespace EnumCompareTests {

enum class EnumA { A, A2 };
enum class EnumB { B };
enum class EnumC : unsigned { C };

void test_enum_enum_compare_no_builtin() {
  auto r1 = (EnumA::A <=> EnumA::A2); // OK
  ASSERT_EXPR_TYPE(r1, std::strong_ordering);
  (void)(EnumA::A <=> EnumA::A); // expected-warning {{self-comparison always evaluates to 'std::strong_ordering::equal'}}
  (void)(EnumA::A <=> EnumB::B); // expected-error {{invalid operands to binary expression ('EnumCompareTests::EnumA' and 'EnumCompareTests::EnumB')}}
  (void)(EnumB::B <=> EnumA::A); // expected-error {{invalid operands}}
}

template <int>
struct Tag {};
Tag<0> operator<=>(EnumA, EnumA) { // expected-note {{not viable}}
  return {};
}
// expected-note@+1 {{while rewriting comparison as call to 'operator<=>' declared here}}
Tag<1> operator<=>(EnumA, EnumB) { // expected-note {{not viable}}
  return {};
}

void test_enum_ovl_provided() {
  auto r1 = (EnumA::A <=> EnumA::A);
  ASSERT_EXPR_TYPE(r1, Tag<0>);
  auto r2 = (EnumA::A <=> EnumB::B);
  ASSERT_EXPR_TYPE(r2, Tag<1>);
  (void)(EnumB::B <=> EnumA::A); // expected-error {{invalid operands to binary expression ('int' and 'Tag<1>')}}
}

void enum_float_test() {
  enum EnumA { A };
  (void)(A <=> (float)0);       // expected-error {{invalid operands to binary expression ('EnumA' and 'float')}}
  (void)((double)0 <=> A);      // expected-error {{invalid operands to binary expression ('double' and 'EnumA')}}
  (void)((long double)0 <=> A); // expected-error {{invalid operands to binary expression ('long double' and 'EnumA')}}
}

enum class Bool1 : bool { Zero,
                          One };
enum Bool2 : bool { B2_Zero,
                    B2_One };

void test_bool_enum(Bool1 A1, Bool1 A2, Bool2 B1, Bool2 B2) {
  (void)(A1 <=> A2);
  (void)(B1 <=> B2);
}

} // namespace EnumCompareTests

namespace TestUserDefinedConvSeq {

template <class T, T Val>
struct Conv {
  constexpr operator T() const { return Val; }
  operator T() { return Val; }
};

void test_user_conv() {
  {
    using C = Conv<int, 0>;
    C c;
    const C cc;
    (void)(0 <=> c);
    (void)(c <=> -1);
    (void)((unsigned)0 <=> cc);
    (void)((unsigned)0 <=> c); // expected-error {{argument to 'operator<=>' cannot be narrowed from type 'int' to 'unsigned int'}}
  }
  {
    using C = Conv<int, -1>;
    C c;
    const C cc;
    (void)(c <=> 0);
    (void)(cc <=> (unsigned)0); // expected-error {{argument to 'operator<=>' evaluates to -1, which cannot be narrowed to type 'unsigned int'}}
    (void)(c <=> (unsigned)0);  // expected-error {{cannot be narrowed from type 'int' to 'unsigned int'}}
  }
}

struct X {
  constexpr const Conv<int, -1> operator<=>(X) { return {}; }
};
static_assert(X() < X());

} // namespace TestUserDefinedConvSeq

void test_array_conv() {
  int arr[5];
  int *ap = arr + 2;
  int arr2[3];
  (void)(arr <=> arr); // expected-error {{invalid operands to binary expression ('int[5]' and 'int[5]')}}
  (void)(+arr <=> arr);
}

void test_mixed_float_int(float f, double d, long double ld) {
  extern int i;
  extern unsigned u;
  extern long l;
  extern short s;
  extern unsigned short us;
  auto r1 = (f <=> i);
  ASSERT_EXPR_TYPE(r1, std::partial_ordering);

  auto r2 = (us <=> ld);
  ASSERT_EXPR_TYPE(r2, std::partial_ordering);

  auto r3 = (s <=> f);
  ASSERT_EXPR_TYPE(r3, std::partial_ordering);

  auto r4 = (0.0 <=> i);
  ASSERT_EXPR_TYPE(r4, std::partial_ordering);
}

namespace NullptrTest {
using nullptr_t = decltype(nullptr);
void foo(nullptr_t x, nullptr_t y) {
  auto r = x <=> y; // expected-error {{invalid operands}}
}
} // namespace NullptrTest

namespace ComplexTest {

enum class StrongE {};
enum WeakE { E_One,
             E_Two };

void test_diag(_Complex int ci, _Complex float cf, _Complex double cd, int i, float f, StrongE E1, WeakE E2, int *p) {  // expected-warning 3 {{'_Complex' is a C99 extension}}
  (void)(ci <=> (_Complex int &)ci); // expected-warning {{'_Complex' is a C99 extension}} expected-error {{invalid operands}}
  (void)(ci <=> cf); // expected-error {{invalid operands}}
  (void)(ci <=> i); // expected-error {{invalid operands}}
  (void)(ci <=> f); // expected-error {{invalid operands}}
  (void)(cf <=> i); // expected-error {{invalid operands}}
  (void)(cf <=> f); // expected-error {{invalid operands}}
  (void)(ci <=> p); // expected-error {{invalid operands}}
  (void)(ci <=> E1); // expected-error {{invalid operands}}
  (void)(E2 <=> cf); // expected-error {{invalid operands}}
}

void test_int(_Complex int x, _Complex int y) { // expected-warning 2 {{'_Complex' is a C99 extension}}
  auto r = x <=> y; // expected-error {{invalid operands}}
}

void test_double(_Complex double x, _Complex double y) { // expected-warning 2 {{'_Complex' is a C99 extension}}
  auto r = x <=> y; // expected-error {{invalid operands}}
}

} // namespace ComplexTest

namespace Vector {
  typedef __attribute__((ext_vector_type(4))) int V;
  void f(V v1, V v2) {
    // This would logically result in a vector of std::strong_ordering, but we
    // don't support vectors of class type. We could model this as a vector of
    // int (-1 / 0 / 1), but that doesn't extend to floating-point types (how
    // to represent 'unordered')? For now, just reject.
    (void)(v1 <=> v2); // expected-error {{three-way comparison between vectors is not supported}}
  }
}

namespace PR44992 {
  extern "C++" struct s {
    friend auto operator<=>(s const &, s const &) = default;
  };
}

namespace PR52537 {
  template<typename T> struct X {};
  template<typename T> bool operator==(const X<T> &, int) { return T::error; } // expected-error 2{{no members}}
  template<typename T> int operator<=>(const X<T> &, int) { return T::error; } // expected-error 2{{no members}}

  const X<int[1]> x1;
  template<typename T> bool f1() { return x1 != 0; } // expected-note {{instantiation of}}
  void g1() { f1<int>(); } // expected-note {{instantiation of}}

  const X<int[2]> x2;
  template<typename T> bool f2() { return 0 == x2; } // expected-note {{instantiation of}}
  void g2() { f2<int>(); } // expected-note {{instantiation of}}

  const X<int[3]> x3;
  template<typename T> bool f3() { return x3 < 0; } // expected-note {{instantiation of}}
  void g3() { f3<int>(); } // expected-note {{instantiation of}}

  const X<int[4]> x4;
  template<typename T> bool f4() { return 0 >= x4; } // expected-note {{instantiation of}}
  void g4() { f4<int>(); } // expected-note {{instantiation of}}

  template<typename T> struct Y {};
  template<typename T> struct Z { Z(int) { T::error; } using nondeduced = Z; }; // expected-error 2{{no members}}
  template<typename T> Z<T> operator<=>(const Y<T>&, int);
  template<typename T> bool operator<(const Z<T>&, const typename Z<T>::nondeduced&);
  template<typename T> bool operator<(const typename Z<T>::nondeduced&, const Z<T>&);

  const Y<int[5]> y5;
  template<typename T> bool f5() { return y5 < 0; } // expected-note {{instantiation of}}
  void g5() { f5<int>(); } // expected-note {{instantiation of}}

  const Y<int[6]> y6;
  template<typename T> bool f6() { return 0 < y6; } // expected-note {{instantiation of}}
  void g6() { f6<int>(); } // expected-note {{instantiation of}}
}


namespace GH64923 {
using nullptr_t = decltype(nullptr);
struct MyTask{};
constexpr MyTask DoAnotherThing() {
    return {};
}

constexpr nullptr_t operator++(MyTask &&T); // expected-note 2{{declared here}}
void DoSomething() {
  if constexpr (++DoAnotherThing() != nullptr) {} // expected-error {{constexpr if condition is not a constant expression}} \
                                                  // expected-note  {{undefined function 'operator++' cannot be used in a constant expression}}

  if constexpr (nullptr == ++DoAnotherThing()) {} // expected-error {{constexpr if condition is not a constant expression}} \
                                                  // expected-note  {{undefined function 'operator++' cannot be used in a constant expression}}
}
}
