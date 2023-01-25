// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify

template <int N> struct Str {
  constexpr Str(char const (&s)[N]) { __builtin_memcpy(value, s, N); }
  char value[N];
};

template <Str V> class ASCII {};

void Foo(ASCII<"this nontype template argument is too long to print">); // expected-note {{no known conversion from 'ASCII<{"this nontype template argument is too long"}>' to 'ASCII<{"this nontype template argument is too long to print"}>'}}
void Bar(ASCII<"this nttp argument is too short">);                     // expected-note {{no known conversion from 'ASCII<{{119, 97, 105, 116, 32, 97, 32, 115, 27, 99, 111, 110, 100, 0}}>' to 'ASCII<{"this nttp argument is too short"}>'}}
void Meow(ASCII<"what|">);                                              // expected-note {{no known conversion from 'ASCII<{"what??!"}>' to 'ASCII<{"what|"}>' for 1st argument}}

void test_ascii() {
  ASCII<"this nontype template argument"
        " is too long">
      a;
  Foo(a);                 // expected-error {{no matching function}}
  decltype(a)::display(); // expected-error {{no member named 'display' in 'ASCII<Str<43>{"this nontype template argument is [...]"}>'}}
}

void test_non_ascii() {
  ASCII<"wait a s\033cond"> a;
  Bar(a);                 // expected-error {{no matching function}}
  decltype(a)::display(); // expected-error {{no member named 'display' in 'ASCII<Str<14>{{119, 97, 105, 116, 32, 97, 32, 115, 27, 99, ...}}>'}}
}

// The dialects (C++20 and above) that accept string literals as non-type
// template arguments do not support trigraphs.
void test_trigraph() {
  ASCII<"what??!"> a;     // expected-warning {{trigraph ignored}}
  Meow(a);                // expected-error {{no matching function}}
  decltype(a)::display(); // expected-error {{no member named 'display' in 'ASCII<Str<8>{"what??!"}>'}}
}
