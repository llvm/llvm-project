// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++26 -freflection -fexperimental-new-constant-interpreter -fsyntax-only -verify

#include <meta>

static_assert(__is_same(std::meta::info, decltype(^^int)));

struct S {};
namespace A{};

consteval void test_pass() {
  static_assert(std::meta::is_type(^^int));
  static_assert(std::meta::is_type(^^unsigned));
  static_assert(std::meta::is_type(^^long));
  static_assert(std::meta::is_type(^^long long));
  static_assert(std::meta::is_type(^^short));
  static_assert(std::meta::is_type(^^char));
  static_assert(std::meta::is_type(^^signed char));
  static_assert(std::meta::is_type(^^unsigned char));
  static_assert(std::meta::is_type(^^wchar_t));
  static_assert(std::meta::is_type(^^char8_t));
  static_assert(std::meta::is_type(^^char16_t));
  static_assert(std::meta::is_type(^^char32_t));
  static_assert(std::meta::is_type(^^bool));
  static_assert(std::meta::is_type(^^float));
  static_assert(std::meta::is_type(^^double));
  static_assert(std::meta::is_type(^^long double));
  static_assert(std::meta::is_type(^^void));
  static_assert(std::meta::is_type(^^decltype(nullptr)));

  constexpr std::meta::info null{};
  static_assert(!std::meta::is_type(null));
}

using IntAlias = int;
static_assert(std::meta::is_type(^^const int));
static_assert(std::meta::is_type(^^volatile int));
static_assert(std::meta::is_type(^^IntAlias));
static_assert(std::meta::is_type(^^std::meta::info));

constexpr std::meta::info r = ^^int;
static_assert(std::meta::is_type(r));
static_assert(!std::meta::is_type(std::meta::info{}));

template <std::meta::info R> consteval bool f() { return std::meta::is_type(R); }
static_assert(f<^^int>());

bool x = std::meta::is_type(^^int);
constexpr bool y = std::meta::is_type(^^int);

auto *fp = &std::meta::is_type; // expected-error {{cannot take address of consteval function 'is_type' outside of an immediate invocation}}
                                // expected-note@*:* {{declared here}}

consteval void test_fail() {
  static_assert(std::meta::is_type(^^S)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(!std::meta::is_type(^^A)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(std::meta::is_type(^^int*)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(std::meta::is_type(^^int&)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(std::meta::is_type(^^int[3])); // expected-error {{unknown or unimplemented reflectable entity}}
}

consteval bool test_with_no_arg() {
  return std::meta::is_type(); // expected-error {{too few arguments to function call, single argument '__r' was not specified}}
  // expected-note@*:* {{'is_type' declared here}}
}

consteval bool test_with_more_than_args_needed() {
  return std::meta::is_type(^^int, ^^float); // expected-error {{too many arguments to function call, expected single argument '__r', have 2 arguments}}
  // expected-note@*:* {{'is_type' declared here}}
}

consteval bool test_with_wrong_type() {
  return std::meta::is_type(42); // expected-error {{cannot initialize a parameter of type 'info' (aka 'std::meta::info') with an rvalue of type 'int'}}
  // expected-note@*:* {{passing argument to parameter '__r' here}}
}

