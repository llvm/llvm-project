// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify -Wno-objc-root-class %s

#if __has_feature(objc_c_static_assert)
#error failed
#endif
#if !__has_extension(objc_c_static_assert)
#error failed
#endif

#if __cplusplus >= 201103L

#if !__has_feature(objc_cxx_static_assert)
#error failed
#endif

// C++11

@interface A {
  int a;
  static_assert(1, "");
  _Static_assert(1, "");

  static_assert(0, ""); // expected-error {{static_assert failed}}
  _Static_assert(0, ""); // expected-error {{static_assert failed}}

  static_assert(a, ""); // expected-error {{static_assert expression is not an integral constant expression}}
  static_assert(sizeof(a) == 4, "");
  static_assert(sizeof(a) == 3, ""); // expected-error {{static_assert failed}}
}

static_assert(1, "");
_Static_assert(1, "");

- (void)f;
@end

@implementation A {
  int b;
  static_assert(1, "");
  _Static_assert(1, "");
  static_assert(sizeof(b) == 4, "");
  static_assert(sizeof(b) == 3, ""); // expected-error {{static_assert failed}}
}

static_assert(1, "");

- (void)f {
  static_assert(1, "");
}
@end

@interface B
@end

@interface B () {
  int b;
  static_assert(sizeof(b) == 4, "");
  static_assert(sizeof(b) == 3, ""); // expected-error {{static_assert failed}}
}
@end

#else

#if __has_feature(objc_cxx_static_assert)
#error failed
#endif

// C++98
@interface A {
  int a;
  static_assert(1, ""); // expected-error {{type name requires a specifier or qualifier}} expected-error{{expected parameter declarator}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  _Static_assert(1, "");
  _Static_assert(0, ""); // expected-error {{static_assert failed}}
}
@end
#endif
