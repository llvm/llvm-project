// RUN: %clang_cc1 %s -triple x86_64-unknown-linux -std=c++26 -freflection -fsyntax-only -verify=expected,std
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux -std=c++26 -freflection -fexperimental-new-constant-interpreter -fsyntax-only -verify=expected,std
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux -std=c++26 -freflection -fsyntax-only -verify=expected,builtin -DUSE_BUILTIN
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux -std=c++26 -freflection -fexperimental-new-constant-interpreter -fsyntax-only -verify=expected,builtin -DUSE_BUILTIN

#include <meta>
#include <stdint.h>
#include <stddef.h>

#ifdef USE_BUILTIN
#  define meta_is_type __builtin_meta_is_type
#else
#  define meta_is_type std::meta::is_type
#endif

static_assert(__is_same(std::meta::info, decltype(^^int)));

struct S {};
namespace A{};

using IntAlias = int;
using ConstIntAlias = const int;
template <typename T> using Identity = T;

consteval void test_pass() {
  static_assert(meta_is_type(^^std::meta::info));
  static_assert(meta_is_type(^^void));
  static_assert(meta_is_type(^^bool));
  static_assert(meta_is_type(^^char));
  static_assert(meta_is_type(^^signed char));
  static_assert(meta_is_type(^^unsigned char));
  static_assert(meta_is_type(^^short));
  static_assert(meta_is_type(^^unsigned short));
  static_assert(meta_is_type(^^int));
  static_assert(meta_is_type(^^unsigned int));
  static_assert(meta_is_type(^^double));
  static_assert(meta_is_type(^^float));
  static_assert(meta_is_type(^^long));
  static_assert(meta_is_type(^^long long));
  static_assert(meta_is_type(^^unsigned long));
  static_assert(meta_is_type(^^unsigned long long));

  static_assert(meta_is_type(^^char8_t));
  static_assert(meta_is_type(^^char16_t));
  static_assert(meta_is_type(^^char32_t));
  static_assert(meta_is_type(^^wchar_t));
  static_assert(meta_is_type(^^int8_t));
  static_assert(meta_is_type(^^int16_t));
  static_assert(meta_is_type(^^int32_t));
  static_assert(meta_is_type(^^int64_t));
  static_assert(meta_is_type(^^uint8_t));
  static_assert(meta_is_type(^^uint16_t));
  static_assert(meta_is_type(^^uint32_t));
  static_assert(meta_is_type(^^uint64_t));
  static_assert(meta_is_type(^^intmax_t));
  static_assert(meta_is_type(^^uintmax_t));
  static_assert(meta_is_type(^^intptr_t));
  static_assert(meta_is_type(^^uintptr_t));
  static_assert(meta_is_type(^^size_t));
  static_assert(meta_is_type(^^__int128_t));
  static_assert(meta_is_type(^^__uint128_t));

  static_assert(meta_is_type(^^const int));
  static_assert(meta_is_type(^^volatile int));
  static_assert(meta_is_type(^^const volatile int));
  static_assert(meta_is_type(^^int const));
  static_assert(meta_is_type(^^const unsigned long long));
  static_assert(meta_is_type(^^volatile double));
  static_assert(meta_is_type(^^const bool));
  static_assert(meta_is_type(^^const std::meta::info));

  static_assert(meta_is_type(^^IntAlias));
  static_assert(meta_is_type(^^const IntAlias));
  static_assert(meta_is_type(^^ConstIntAlias));
  static_assert(meta_is_type(^^volatile ConstIntAlias));
  static_assert(meta_is_type(^^const int32_t));
  static_assert(meta_is_type(^^Identity<int>));
  static_assert(meta_is_type(^^Identity<const float>));

  static_assert(meta_is_type(^^decltype(nullptr)));
  static_assert(meta_is_type(^^decltype(42)));
  static_assert(meta_is_type(^^decltype(sizeof(int))));
  static_assert(meta_is_type(^^decltype(^^int)));

  constexpr std::meta::info null{};
  static_assert(!meta_is_type(null));
}


constexpr std::meta::info r = ^^int;
static_assert(meta_is_type(r));
static_assert(!meta_is_type(std::meta::info{}));

template <std::meta::info R> consteval bool f() { return meta_is_type(R); }
static_assert(f<^^int>());

bool x = meta_is_type(^^int);
constexpr bool y = meta_is_type(^^int);

auto *fp = &meta_is_type;
// std-error@-1 {{cannot take address of consteval function 'is_type' outside of an immediate invocation}}
// std-note@*:* {{declared here}}
// builtin-error@-3 {{builtin functions must be directly called}}
// builtin-error@-4 {{cannot take address of consteval function '__builtin_meta_is_type' outside of an immediate invocation}}

consteval void test_fail() {
  static_assert(meta_is_type(^^S)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(!meta_is_type(^^A)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(meta_is_type(^^int*)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(meta_is_type(^^int&)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(meta_is_type(^^int[3])); // expected-error {{unknown or unimplemented reflectable entity}}
}

consteval bool test_with_no_arg() {
  return meta_is_type();
  // std-error@-1 {{too few arguments to function call, single argument '__r' was not specified}}
  // std-note@*:* {{'is_type' declared here}}
  // builtin-error@-3 {{too few arguments to function call, expected 1, have 0}}
}

consteval bool test_with_more_than_args_needed() {
  return meta_is_type(^^int, ^^float);
  // std-error@-1 {{too many arguments to function call, expected single argument '__r', have 2 arguments}}
  // std-note@*:* {{'is_type' declared here}}
  // builtin-error@-3 {{too many arguments to function call, expected 1, have 2}}
}

consteval bool test_with_wrong_type() {
  return meta_is_type(42);
  // std-error@-1 {{cannot initialize a parameter of type 'info' (aka 'std::meta::info') with an rvalue of type 'int'}}
  // std-note@*:* {{passing argument to parameter '__r' here}}
  // builtin-error@-3 {{cannot initialize a parameter of type 'std::meta::info' with an rvalue of type 'int'}}
}
