// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++26 -freflection -fexperimental-new-constant-interpreter -fsyntax-only -verify

using info = decltype(^^int);

struct S {};
namespace A{};

consteval void test_pass() {
  static_assert(__builtin_meta_is_type(^^int));
  static_assert(__builtin_meta_is_type(^^unsigned));
  static_assert(__builtin_meta_is_type(^^long));
  static_assert(__builtin_meta_is_type(^^long long));
  static_assert(__builtin_meta_is_type(^^short));
  static_assert(__builtin_meta_is_type(^^char));
  static_assert(__builtin_meta_is_type(^^signed char));
  static_assert(__builtin_meta_is_type(^^unsigned char));
  static_assert(__builtin_meta_is_type(^^wchar_t));
  static_assert(__builtin_meta_is_type(^^char8_t));
  static_assert(__builtin_meta_is_type(^^char16_t));
  static_assert(__builtin_meta_is_type(^^char32_t));
  static_assert(__builtin_meta_is_type(^^bool));
  static_assert(__builtin_meta_is_type(^^float));
  static_assert(__builtin_meta_is_type(^^double));
  static_assert(__builtin_meta_is_type(^^long double));
  static_assert(__builtin_meta_is_type(^^void));
  static_assert(__builtin_meta_is_type(^^decltype(nullptr)));

  constexpr info null{};
  static_assert(!__builtin_meta_is_type(null));
}

consteval void test_fail() {
  static_assert(__builtin_meta_is_type(^^S)); // expected-error {{unknown or unimplemented reflectable entity}}
  static_assert(!__builtin_meta_is_type(^^A)); // expected-error {{unknown or unimplemented reflectable entity}}
}

consteval bool test_with_no_arg() {
  return __builtin_meta_is_type(); // expected-error {{too few arguments to function call, expected 1, have 0}}
}

consteval bool test_with_more_than_args_needed() {
  return __builtin_meta_is_type(^^int, ^^float); // expected-error {{too many arguments to function call, expected 1, have 2}}
}

consteval bool test_with_wrong_type() {
  return __builtin_meta_is_type(42); // expected-error {{cannot initialize a parameter of type 'std::meta::info' with an rvalue of type 'int'}}
}
