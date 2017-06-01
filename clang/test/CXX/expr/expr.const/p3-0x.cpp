// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++1z -verify %s

// A converted constant expression of type T is a core constant expression,
int nonconst = 8; // expected-note 3 {{here}}
enum NonConstE : unsigned char { NCE = nonconst }; // expected-error {{enumerator value is not a constant expression}} expected-note {{read of non-const}}
template<int = nonconst> struct NonConstT {}; // expected-error {{non-type template argument is not a constant expression}} expected-note {{read of non-const}}
void NonConstF() {
  switch (nonconst) {
    case nonconst: // expected-error {{case value is not a constant expression}} expected-note {{read of non-const}}
      break;
  }
  return;
}

// implicitly converted to a prvalue of type T, where the converted expression
// is a literal constant expression

bool a(int n) {
  constexpr char vowels[] = "aeiou";
  switch (n) {
  case vowels[0]:
  case vowels[1]:
  case vowels[2]:
  case vowels[3]:
  case vowels[4]:
    static_assert(!vowels[5], "unexpected number of vowels");
    return true;
  }
  return false;
}

// and the implicit conversion sequence contains only
//
//  user-defined conversions,
struct S { constexpr operator int() const { return 5; } };
enum E : unsigned char { E5 = S(), E6, E10 = S() * 2, E1 = E5 / 5 };

//  lvalue-to-rvalue conversions,
const E e10 = E10;
template<E> struct T {};
T<e10> s10;

//  integral promotions,
enum class EE { EE32 = ' ', EE65 = 'A', EE1 = (short)1, EE5 = E5 };

//  integral conversions other than narrowing conversions,
int b(unsigned n) {
  switch (n) {
    case E6:
    case EE::EE32: // expected-error {{not implicitly convertible}}
    case (int)EE::EE32:
    case 1000:
    case (long long)1e10: // expected-error {{case value evaluates to 10000000000, which cannot be narrowed to type 'unsigned int'}}
    case -3: // expected-error {{case value evaluates to -3, which cannot be narrowed to type 'unsigned int'}}
      return n;
  }
  return 0;
}
enum class EEE : unsigned short {
  a = E6,
  b = EE::EE32, // expected-error {{not implicitly convertible}}
  c = (int)EE::EE32,
  d = 1000,
  e = 123456, // expected-error {{enumerator value evaluates to 123456, which cannot be narrowed to type 'unsigned short'}}
  f = -3 // expected-error {{enumerator value evaluates to -3, which cannot be narrowed to type 'unsigned short'}}
};
template<unsigned char> using A = int;
using Int = A<E6>;
using Int = A<EE::EE32>; // expected-error {{not implicitly convertible}}
using Int = A<(int)EE::EE32>;
using Int = A<200>;
using Int = A<1000>; // expected-error {{template argument evaluates to 1000, which cannot be narrowed to type 'unsigned char'}}
using Int = A<-3>; // expected-error {{template argument evaluates to -3, which cannot be narrowed to type 'unsigned char'}}

// Note, conversions from integral or unscoped enumeration types to bool are
// integral conversions as well as boolean conversions.
// FIXME: Per core issue 1407, this is not correct.
template<typename T, T v> struct Val { static constexpr T value = v; };
static_assert(Val<bool, E1>::value == 1, ""); // ok
static_assert(Val<bool, '\0'>::value == 0, ""); // ok
static_assert(Val<bool, U'\1'>::value == 1, ""); // ok
static_assert(Val<bool, E5>::value == 1, ""); // expected-error {{5, which cannot be narrowed to type 'bool'}}

//  function pointer conversions [C++17]
void noexcept_false() noexcept(false);
void noexcept_true() noexcept(true);
Val<decltype(&noexcept_false), &noexcept_true> remove_noexcept;
Val<decltype(&noexcept_true), &noexcept_false> add_noexcept;
#if __cplusplus > 201402L
// expected-error@-2 {{value of type 'void (*)() noexcept(false)' is not implicitly convertible to 'void (*)() noexcept'}}
#endif

// (no other conversions are permitted)
using Int = A<1.0>; // expected-error {{conversion from 'double' to 'unsigned char' is not allowed in a converted constant expression}}
enum B : bool {
  True = &a, // expected-error {{conversion from 'bool (*)(int)' to 'bool' is not allowed in a converted constant expression}}
  False = nullptr // expected-error {{conversion from 'nullptr_t' to 'bool' is not allowed in a converted constant expression}}
};
void c() {
  // Note, promoted type of switch is 'int'.
  switch (bool b = a(5)) { // expected-warning {{boolean value}}
  case 0.0f: // expected-error {{conversion from 'float' to 'int' is not allowed in a converted constant expression}}
    break;
  }
}
template <bool B> int f() { return B; } // expected-note {{candidate template ignored: invalid explicitly-specified argument for template parameter 'B'}}
template int f<&S::operator int>(); // expected-error {{does not refer to a function template}}
template int f<(bool)&S::operator int>();

int n = Val<bool, &S::operator int>::value; // expected-error-re {{conversion from 'int (S::*)(){{( __attribute__\(\(thiscall\)\))?}} const' to 'bool' is not allowed in a converted constant expression}}

namespace NonConstLValue {
  struct S {
    constexpr operator int() const { return 10; }
  };
  S s; // not constexpr
  // Under the FDIS, this is not a converted constant expression.
  // Under the new proposed wording, it is.
  enum E : char { e = s };
}
