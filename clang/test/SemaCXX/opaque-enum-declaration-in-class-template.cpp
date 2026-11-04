// RUN: %clang_cc1 -std=c++11 -Wredeclared-class-member -Wconstant-conversion -Wdeprecated-declarations -Wc++11-narrowing -fsyntax-only %s -verify
// RUN: %clang_cc1 -std=c++14 -Wredeclared-class-member -Wconstant-conversion -Wdeprecated-declarations -Wc++11-narrowing -fsyntax-only %s -verify
// RUN: %clang_cc1 -std=c++20 -Wredeclared-class-member -Wconstant-conversion -Wdeprecated-declarations -Wc++11-narrowing -fsyntax-only %s -verify

// Test that opaque-enum-declarations are handled correctly w.r.t integral promotions.
// The key sections in the C++11 standard are:
// C++11 [dcl.enum]p3: An enumeration declared by an opaque-enum-declaration
// has a fixed underlying type and is a complete type.
// C++11 [conv.prom]: A prvalue of an unscoped enumeration type whose underlying type
// is fixed ([dcl.enum]) can be converted to a prvalue of its underlying type.

// This program causes clang 19 and earlier to crash because
// EnumDecl::PromotionType has not been set on the instantiated enum.
// See GitHub Issue #117960.
namespace Issue117960 {
template <typename T>
struct A {
  enum E : T;
};

int b = A<int>::E{} + 0;
}


namespace test {
template <typename T1, typename T2>
struct IsSame {
  static constexpr bool check() { return false; }
};

template <typename T>
struct IsSame<T, T> {
  static constexpr bool check() { return true; }
};
}  // namespace test


template <typename T>
struct S1 {
  enum E : T;
};
// checks if EnumDecl::PromotionType is set
int X1 = S1<int>::E{} + 0;
int Y1 = S1<unsigned>::E{} + 0;
static_assert(test::IsSame<decltype(S1<int>::E{}+0), int>::check(), "");
static_assert(test::IsSame<decltype(S1<unsigned>::E{}+0), unsigned>::check(), "");
char Z1 = S1<unsigned>::E(-1) + 0; // expected-warning{{implicit conversion from 'unsigned int' to 'char'}}

template <typename Traits>
struct S2 {
  enum E : typename Traits::IntegerType;
};

template <typename T>
struct Traits {
  typedef T IntegerType;
};

int X2 = S2<Traits<int>>::E{} + 0;
int Y2 = S2<Traits<unsigned>>::E{} + 0;
static_assert(test::IsSame<decltype(S2<Traits<int>>::E{}+0), int>::check(), "");
static_assert(test::IsSame<decltype(S2<Traits<unsigned>>::E{}+0), unsigned>::check(), "");
// C++11 [conv.prom]p4:
// A prvalue of an unscoped enumeration type whose underlying type is fixed can be converted to a
// prvalue of its underlying type. Moreover, if integral promotion can be applied to its underlying type, a
// prvalue of an unscoped enumeration type whose underlying type is fixed can also be converted to a prvalue
// of the promoted underlying type.
static_assert(test::IsSame<decltype(S2<Traits<char>>::E{}+char(0)), int>::check(), "");


template <typename T>
struct S3 {
  enum E : unsigned;
};

int X3 = S3<float>::E{} + 0;

// fails in clang 19 and earlier (see the discussion on GitHub Issue #117960):
static_assert(test::IsSame<decltype(S3<float>::E{}+0), unsigned>::check(), "");

template <typename T>
struct S4 {
  enum E1 : char;
  enum E2 : T;
};

int X4 = S4<char>::E1{} + '\0';
int Y4 = S4<char>::E2{} + '\0';

template <typename T>
struct S5 {
  enum class E1 : char;
  enum class E2 : T;
};

int X5 = S5<char>::E1{} + '\0'; // expected-error{{invalid operands to binary expression}}
                                // expected-note@-1{{no implicit conversion for scoped enum; consider casting to underlying type}}
int Y5 = S5<char>::E2{} + '\0'; // expected-error{{invalid operands to binary expression}}
                                // expected-note@-1{{no implicit conversion for scoped enum; consider casting to underlying type}}


template <typename T>
struct S6 {
  enum E1 : T;
  enum E2 : E1; // expected-error{{invalid underlying type}}
};

template struct S6<int>; // expected-note{{in instantiation of template class 'S6<int>' requested here}}


template <typename T>
struct S7 {
  enum E : T;
  enum E : T { X, Y, Z }; // expected-note{{previous declaration is here}}
  enum E : T; // expected-warning{{class member cannot be redeclared}}
};

template struct S7<int>;

template <typename T>
struct S8 {
  enum E : char;
  enum E : char { X, Y, Z }; // expected-note{{previous declaration is here}}
  enum E : char; // expected-warning{{class member cannot be redeclared}}
};

template struct S8<float>;

template <typename T>
struct S9 {
  enum class E1 : T;
  enum class E1 : T { X, Y, Z }; // expected-note{{previous declaration is here}}
  enum class E1 : T; // expected-warning{{class member cannot be redeclared}}
  enum class E2 : char;
  enum class E2 : char { X, Y, Z }; // expected-note{{previous declaration is here}}
  enum class E2 : char; // expected-warning{{class member cannot be redeclared}}
};

template struct S9<int>;

#if defined(__cplusplus) && __cplusplus >= 201402L
template <typename T>
struct S10 {
  enum [[deprecated("for reasons")]] E : T; // expected-note{{explicitly marked deprecated here}}
};

int X10 = S10<int>::E{} + 0; // expected-warning{{deprecated: for reasons}}
#endif

template <typename T>
struct S11 {};

template <>
struct S11<unsigned> {
  enum E : unsigned;
};

unsigned X11 = S11<unsigned>::E{} + 0u;

#if defined(__cplusplus) && __cplusplus >= 201402L
template <typename T>
struct S12 {
  enum [[deprecated("for reasons")]] E1 : T; // expected-note{{explicitly marked deprecated here}}
  enum [[deprecated("for reasons")]] E2 : T;
};

template <>
struct S12<float> {
  enum E1 : unsigned;
  enum E2 : unsigned;
};

unsigned X12 = S12<float>::E1{} + 0u;
unsigned Y12 = S12<float>::E2{} + 0u;
int Z12 = S12<int>::E1{} + 0; // expected-warning{{deprecated: for reasons}}
#endif

template <typename T>
struct S13 {
  enum __attribute__((packed)) E { X, Y };
};

static_assert(sizeof(S13<int>::E) == 1, "");

template<typename T>
struct S14 {
  enum E : float; // expected-error {{invalid underlying type}}
};

template<typename T>
struct S15 {
  enum E : T; // expected-error {{invalid underlying type}}
};

template struct S15<float>; // expected-note {{in instantiation of template class 'S15<float>' requested here}}




template <typename T>
int f1() {
  enum E : T;
  return E{} + 0;
}

int F1 = f1<int>();

template <typename T>
int f2() {
  struct LocalClass {
    enum E : T;
  };
  return typename LocalClass::E{} + 0;
}

int F2 = f2<int>();
