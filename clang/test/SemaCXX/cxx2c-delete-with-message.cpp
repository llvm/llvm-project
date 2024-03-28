// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

struct S {
  void f() = delete("deleted (1)"); // expected-note {{explicitly marked deleted}}

  template <typename T>
  T g() = delete("deleted (2)"); // expected-note {{explicitly deleted}}
};

template <typename T>
struct TS {
  T f() = delete("deleted (3)"); // expected-note {{explicitly marked deleted}}

  template <typename U>
  T g(U) = delete("deleted (4)"); // expected-note {{explicitly deleted}}
};

void f() = delete("deleted (5)"); // expected-note {{explicitly deleted}}

template <typename T>
T g() = delete("deleted (6)"); // expected-note {{explicitly deleted}}

void u1() = delete(L"\xFFFFFFFF"); // expected-error {{an unevaluated string literal cannot have an encoding prefix}} \
                                   // expected-error {{invalid escape sequence '\xFFFFFFFF' in an unevaluated string literal}}
void u2() = delete(u"\U000317FF"); // expected-error {{an unevaluated string literal cannot have an encoding prefix}}

void u3() = delete("Ω"); // expected-note {{explicitly deleted}}
void u4() = delete("\u1234"); // expected-note {{explicitly deleted}}

void u5() = delete("\x1ff"       // expected-error {{hex escape sequence out of range}} \
                                 // expected-error {{invalid escape sequence '\x1ff' in an unevaluated string literal}}
                     "0\x123"    // expected-error {{invalid escape sequence '\x123' in an unevaluated string literal}}
                     "fx\xfffff" // expected-error {{invalid escape sequence '\xfffff' in an unevaluated string literal}}
                     "goop");

void u6() = delete("\'\"\?\\\a\b\f\n\r\t\v"); // expected-note {{explicitly deleted}}
void u7() = delete("\xFF"); // expected-error {{invalid escape sequence '\xFF' in an unevaluated string literal}}
void u8() = delete("\123"); // expected-error {{invalid escape sequence '\123' in an unevaluated string literal}}
void u9() = delete("\pOh no, a Pascal string!"); // expected-warning {{unknown escape sequence '\p'}} \
                                                 // expected-error {{invalid escape sequence '\p' in an unevaluated string literal}}
// expected-note@+1 {{explicitly deleted}}
void u10() = delete(R"(a
\tb
c
)");

void u11() = delete("\u0080\u0081\u0082\u0083\u0099\u009A\u009B\u009C\u009D\u009E\u009F"); // expected-note {{explicitly deleted}}


//! Contains RTL/LTR marks
void u12() = delete("\u200Eabc\u200Fdef\u200Fgh"); // expected-note {{explicitly deleted}}

//! Contains ZWJ/regional indicators
void u13() = delete("🏳️‍🌈 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🇪🇺"); // expected-note {{explicitly deleted}}

void h() {
  S{}.f(); // expected-error {{attempt to use a deleted function: deleted (1)}}
  S{}.g<int>(); // expected-error {{call to deleted member function 'g': deleted (2)}}
  TS<int>{}.f(); // expected-error {{attempt to use a deleted function: deleted (3)}}
  TS<int>{}.g<int>(0); // expected-error {{call to deleted member function 'g': deleted (4)}}
  f(); // expected-error {{call to deleted function 'f': deleted (5)}}
  g<int>(); // expected-error {{call to deleted function 'g': deleted (6)}}
  u3(); // expected-error {{call to deleted function 'u3': Ω}}
  u4(); // expected-error {{call to deleted function 'u4': ሴ}}
  u6(); // expected-error {{call to deleted function 'u6': '"?\<U+0007><U+0008>}}
  u10(); // expected-error {{call to deleted function 'u10': a\n\tb\nc\n}}
  u11(); // expected-error {{call to deleted function 'u11': <U+0080><U+0081><U+0082><U+0083><U+0099><U+009A><U+009B><U+009C><U+009D><U+009E><U+009F>}}
  u12(); // expected-error {{call to deleted function 'u12': ‎abc‏def‏gh}}
  u13(); // expected-error {{call to deleted function 'u13': 🏳️‍🌈 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🇪🇺}}
}
