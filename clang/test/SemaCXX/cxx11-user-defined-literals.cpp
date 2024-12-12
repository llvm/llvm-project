// RUN: %clang_cc1 -std=c++11 -verify %s -fms-extensions -triple x86_64-apple-darwin9.0.0

using size_t = decltype(sizeof(int));
enum class LitKind {
  Char, WideChar, Char16, Char32,
  CharStr, WideStr, Char16Str, Char32Str,
  Integer, Floating, Raw, Template
};
constexpr LitKind operator""_kind(char p) { return LitKind::Char; }
constexpr LitKind operator""_kind(wchar_t p) { return LitKind::WideChar; }
constexpr LitKind operator""_kind(char16_t p) { return LitKind::Char16; }
constexpr LitKind operator""_kind(char32_t p) { return LitKind::Char32; }
constexpr LitKind operator""_kind(const char *p, size_t n) { return LitKind::CharStr; }
constexpr LitKind operator""_kind(const wchar_t *p, size_t n) { return LitKind::WideStr; }
constexpr LitKind operator""_kind(const char16_t *p, size_t n) { return LitKind::Char16Str; }
constexpr LitKind operator""_kind(const char32_t *p, size_t n) { return LitKind::Char32Str; }
constexpr LitKind operator""_kind(unsigned long long n) { return LitKind::Integer; }
constexpr LitKind operator""_kind(long double n) { return LitKind::Floating; }
constexpr LitKind operator""_kind2(const char *p) { return LitKind::Raw; }
template<char ...Cs> constexpr LitKind operator""_kind3() { return LitKind::Template; }

static_assert('x'_kind == LitKind::Char, "");
static_assert(L'x'_kind == LitKind::WideChar, "");
static_assert(u'x'_kind == LitKind::Char16, "");
static_assert(U'x'_kind == LitKind::Char32, "");
static_assert("foo"_kind == LitKind::CharStr, "");
static_assert(u8"foo"_kind == LitKind::CharStr, "");
static_assert(L"foo"_kind == LitKind::WideStr, "");
static_assert(u"foo"_kind == LitKind::Char16Str, "");
static_assert(U"foo"_kind == LitKind::Char32Str, "");
static_assert(194_kind == LitKind::Integer, "");
static_assert(0377_kind == LitKind::Integer, "");
static_assert(0x5ffc_kind == LitKind::Integer, "");
static_assert(.5954_kind == LitKind::Floating, "");
static_assert(1._kind == LitKind::Floating, "");
static_assert(1.e-2_kind == LitKind::Floating, "");
static_assert(4e6_kind == LitKind::Floating, "");
static_assert(4e6_kind2 == LitKind::Raw, "");
static_assert(4e6_kind3 == LitKind::Template, "");

constexpr const char *fractional_digits_impl(const char *p) {
  return *p == '.' ? p + 1 : *p ? fractional_digits_impl(p + 1) : 0;
}
constexpr const char *operator""_fractional_digits(const char *p) {
  return fractional_digits_impl(p) ?: p;
}
constexpr bool streq(const char *p, const char *q) {
  return *p == *q && (!*p || streq(p+1, q+1));
}

static_assert(streq(143.97_fractional_digits, "97"), "");
static_assert(streq(0x786_fractional_digits, "0x786"), "");
static_assert(streq(.4_fractional_digits, "4"), "");
static_assert(streq(4._fractional_digits, ""), "");
static_assert(streq(1e+97_fractional_digits, "1e+97"), "");
static_assert(streq(0377_fractional_digits, "0377"), "");
static_assert(streq(0377.5_fractional_digits, "5"), "");

int operator""_ambiguous(char); // expected-note {{candidate}}
namespace N {
  void *operator""_ambiguous(char); // expected-note {{candidate}}
}
using namespace N;
int k = 'x'_ambiguous; // expected-error {{ambiguous}}

int operator""_deleted(unsigned long long) = delete; // expected-note {{here}}
int m = 42_deleted; // expected-error {{attempt to use a deleted}}

namespace Using {
  namespace M {
    int operator""_using(char);
  }
  int k1 = 'x'_using; // expected-error {{no matching literal operator for call to 'operator""_using'}}

  using M::operator ""_using;
  int k2 = 'x'_using;
}

namespace AmbiguousRawTemplate {
  int operator""_ambig1(const char *); // expected-note {{candidate}}
  template<char...> int operator""_ambig1(); // expected-note {{candidate}}

  int k1 = 123_ambig1; // expected-error {{call to 'operator""_ambig1' is ambiguous}}

  namespace Inner {
    template<char...> int operator""_ambig2(); // expected-note 3{{candidate}}
  }
  int operator""_ambig2(const char *); // expected-note 3{{candidate}}
  using Inner::operator""_ambig2;

  int k2 = 123_ambig2; // expected-error {{call to 'operator""_ambig2' is ambiguous}}

  namespace N {
    using Inner::operator""_ambig2;

    int k3 = 123_ambig2; // ok

    using AmbiguousRawTemplate::operator""_ambig2;

    int k4 = 123_ambig2; // expected-error {{ambiguous}}

    namespace M {

      template<char...> int operator""_ambig2();

      int k5 = 123_ambig2; // ok
    }

    int operator""_ambig2(unsigned long long);

    int k6 = 123_ambig2; // ok
    int k7 = 123._ambig2; // expected-error {{ambiguous}}
  }
}

constexpr unsigned mash(unsigned a) {
 return 0x93ae27b5 * ((a >> 13) | a << 19);
}
template<typename=void> constexpr unsigned hash(unsigned a) { return a; }
template<char C, char...Cs> constexpr unsigned hash(unsigned a) {
 return hash<Cs...>(mash(a ^ mash(C)));
}
template<typename T, T v> struct constant { constexpr static T value = v; };
template<char...Cs> constexpr unsigned operator""_hash() {
  return constant<unsigned, hash<Cs...>(0)>::value;
}
static_assert(0x1234_hash == 0x103eff5e, "");
static_assert(hash<'0', 'x', '1', '2', '3', '4'>(0) == 0x103eff5e, "");

// Functions and literal suffixes go in separate namespaces.
namespace Namespace {
  template<char...> int operator""_x();
  int k = _x(); // expected-error {{undeclared identifier '_x'}}

  int _y(unsigned long long);
  int k2 = 123_y; // expected-error {{no matching literal operator for call to 'operator""_y'}}
}

namespace PR14950 {
  template<...> // expected-error {{expected template parameter}}
  int operator""_b(); // expected-error {{no function template matches function template specialization}}
  int main() { return 0_b; } // expected-error {{no matching literal operator for call to 'operator""_b'}}
}

namespace bad_names {
  template<char...> int operator""_x();

  template<typename T> void f() {
    class T:: // expected-error {{anonymous class}} expected-warning {{does not declare anything}}
        operator // expected-error {{expected identifier}}
            ""_q<'a'>;

    T::template operator""_q<'a'>(); // expected-error {{non-namespace scope 'T::' cannot have a literal operator member}} expected-error +{{}}
    T::template operator""_q<'a'>::X; // expected-error {{non-namespace scope 'T::' cannot have a literal operator member}} expected-error +{{}}
    T::operator""_q<'a'>(); // expected-error {{non-namespace scope 'T::' cannot have a literal operator member}} expected-error +{{}}
    typename T::template operator""_q<'a'> a; // expected-error {{non-namespace scope 'T::' cannot have a literal operator member}} expected-error +{{}}
    typename T::operator""_q(""); // expected-error +{{}} expected-note {{to match}}
    T::operator""_q(""); // expected-error {{non-namespace scope 'T::' cannot have a literal operator member}}

    bad_names::operator""_x<'a', 'b', 'c'>();
  };

  struct S {};
  void g() {
    S::operator""_q(); // expected-error {{non-namespace scope 'S::' cannot have a literal operator member}}
  }
}
