// RUN: %clang_cc1 -std=c++11 -verify %s -fms-extensions -triple x86_64-apple-darwin9.0.0

// A ud-suffix cannot be used on string literals in a whole bunch of contexts:

#include "foo"_bar // expected-error {{expected "FILENAME" or <FILENAME>}}
#line 1 "foo"_bar // expected-error {{user-defined suffix cannot be used here}}
# 1 "foo"_bar 1 // expected-error {{user-defined suffix cannot be used here}}
#ident "foo"_bar // expected-error {{user-defined suffix cannot be used here}}
_Pragma("foo"_bar) // expected-error {{user-defined suffix cannot be used here}}
#pragma comment(lib, "foo"_bar) // expected-error {{user-defined suffix cannot be used here}}
_Pragma("comment(lib, \"foo\"_bar)") // expected-error {{user-defined suffix cannot be used here}}
#pragma message "hi"_there // expected-error {{user-defined suffix cannot be used here}} expected-warning {{hi}}
#pragma push_macro("foo"_bar) // expected-error {{user-defined suffix cannot be used here}}
#if __has_warning("-Wan-island-to-discover"_bar) // expected-error {{user-defined suffix cannot be used here}}
#elif __has_include("foo"_bar) // expected-error {{expected "FILENAME" or <FILENAME>}}
#endif

extern "C++"_x {} // expected-error {{user-defined suffix cannot be used here}}

int f() {
  asm("mov %eax, %rdx"_foo); // expected-error {{user-defined suffix cannot be used here}}
}

static_assert(true, "foo"_bar); // expected-error {{no matching literal operator for call to 'operator""_bar'}}
// expected-warning@-1 {{'static_assert' with a user-generated message is a C++26 extension}}

int cake() __attribute__((availability(macosx, unavailable, message = "is a lie"_x))); // expected-error {{user-defined suffix cannot be used here}}

// A ud-suffix cannot be used on character literals in preprocessor constant
// expressions:
#if 'x'_y - u'x'_z // expected-error 2{{character literal with user-defined suffix cannot be used in preprocessor constant expression}}
#error error
#endif

// A ud-suffix cannot be used on integer literals in preprocessor constant
// expressions:
#if 0_foo // expected-error {{integer literal with user-defined suffix cannot be used in preprocessor constant expression}}
#error error
#endif

// But they can appear in expressions.
constexpr char operator""_id(char c) { return c; }
constexpr wchar_t operator""_id(wchar_t c) { return c; }
constexpr char16_t operator""_id(char16_t c) { return c; }
constexpr char32_t operator""_id(char32_t c) { return c; }

using size_t = decltype(sizeof(int));
constexpr const char operator""_id(const char *p, size_t n) { return *p; }
constexpr const wchar_t operator""_id(const wchar_t *p, size_t n) { return *p; }
constexpr const char16_t operator""_id(const char16_t *p, size_t n) { return *p; }
constexpr const char32_t operator""_id(const char32_t *p, size_t n) { return *p; }

constexpr unsigned long long operator""_id(unsigned long long n) { return n; }
constexpr long double operator""_id(long double d) { return d; }

template<int n> struct S {};
S<"a"_id> sa;
S<L"b"_id> sb;
S<u8"c"_id> sc;
S<u"d"_id> sd;
S<U"e"_id> se;

S<'w'_id> sw;
S<L'x'_id> sx;
S<u'y'_id> sy;
S<U'z'_id> sz;

S<100_id> sn;
S<(int)1.3_id> sf;

void h() {
  (void)"test"_id "test" L"test";
}

// Test source location for suffix is known
const char *p =
  "foo\nbar" R"x(
  erk
  flux
  )x" "eep\x1f"\
_no_such_suffix // expected-error {{'operator""_no_such_suffix'}}
"and a bit more"
"and another suffix"_no_such_suffix;

char c =
  '\x14'\
_no_such_suffix; // expected-error {{'operator""_no_such_suffix'}}

int &r =
1234567\
_no_such_suffix; // expected-error {{'operator""_no_such_suffix'}}

int k =
1234567.89\
_no_such_suffix; // expected-error {{'operator""_no_such_suffix'}}

// Make sure we handle more interesting ways of writing a string literal which
// is "" in translation phase 7.
void operator "\
" _foo(unsigned long long); // ok
// expected-warning@-1{{identifier '_foo' preceded by whitespace in a literal operator declaration is deprecated}}

void operator R"xyzzy()xyzzy" _foo(long double); // ok
// expected-warning@-1{{identifier '_foo' preceded by whitespace in a literal operator declaration is deprecated}}

void operator"" "" R"()" "" _foo(const char *); // ok
// expected-warning@-1{{identifier '_foo' preceded by whitespace}}

void operator ""_no_space(const char *); // ok

// Ensure we diagnose the bad cases.
void operator "\0"_non_empty(const char *); // expected-error {{must be '""'}}
void operator L""_not_char(const char *); // expected-error {{cannot have an encoding prefix}}
void operator "" ""
U"" // expected-error {{cannot have an encoding prefix}}
""_also_not_char(const char *);
void operator "" u8"" "\u0123" "hello"_all_of_the_things ""(const char*); // expected-error {{must be '""'}}

// Make sure we treat UCNs and UTF-8 as equivalent.
int operator""_µs(unsigned long long) {} // expected-note {{previous}}
int hundred_µs = 50_µs + 50_\u00b5s;
int operator""_\u00b5s(unsigned long long) {} // expected-error {{redefinition of 'operator""_µs'}}

int operator""_\U0000212B(long double) {} // expected-note {{previous}}
int hundred_Å = 50.0_Å + 50._\U0000212B;
int operator""_Å(long double) {} // expected-error {{redefinition of 'operator""_Å'}}

int operator""_𐀀(char) {} // expected-note {{previous}}
int 𐀀 = '4'_𐀀 + '2'_\U00010000;
int operator""_\U00010000(char) {} // expected-error {{redefinition of 'operator""_𐀀'}}

// These all declare the same function.
int operator""_℮""_\u212e""_\U0000212e""(const char*, size_t);
int operator""_\u212e""_\U0000212e""_℮""(const char*, size_t);
int operator""_\U0000212e""_℮""_\u212e""(const char*, size_t);

int operator""_\u{212f}(char);
int operator""_\N{SCRIPT SMALL E}(char);

int mix_ucn_utf8 = ""_℮""_\u212e""_\U0000212e"";

void operator""_℮""_ℯ(unsigned long long) {} // expected-error {{differing user-defined suffixes ('_℮' and '_ℯ') in string literal concatenation}}
void operator""_℮""_\u212f(unsigned long long) {} // expected-error {{differing user-defined suffixes ('_℮' and '_ℯ') in string literal concatenation}}
void operator""_\u212e""_ℯ(unsigned long long) {} // expected-error {{differing user-defined suffixes ('_℮' and '_ℯ') in string literal concatenation}}
void operator""_\u212e""_\u212f(unsigned long long) {} // expected-error {{differing user-defined suffixes ('_℮' and '_ℯ') in string literal concatenation}}

void operator""_℮""_℮(unsigned long long) {} // expected-note {{previous}}
void operator""_\u212e""_\u212e(unsigned long long) {} // expected-error {{redefinition}}

#define ¢ *0.01 // expected-error {{macro name must be an identifier}}
constexpr int operator""_¢(long double d) { return d * 100; }  // expected-error {{character <U+00A2> not allowed in an identifier}}
constexpr int operator""_¢(unsigned long long n) { return n; } // expected-error {{character <U+00A2> not allowed in an identifier}}
static_assert(0.02_¢ == 2_¢, "");                              // expected-error 2{{character <U+00A2> not allowed in an identifier}}
