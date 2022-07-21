// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -std=c89 -fobjc-runtime=macosx-fragile -fsyntax-only -verify -Wno-objc-root-class %s


#if __STDC_VERSION__ >= 201112L

#if !__has_feature(objc_c_static_assert)
#error failed
#endif

#if !__has_extension(objc_c_static_assert)
#error failed
#endif

@interface A {
  int a;
  _Static_assert(1, "");
  _Static_assert(0, ""); // expected-error {{static_assert failed}}

  _Static_assert(a, ""); // expected-error {{use of undeclared identifier 'a'}}
  _Static_assert(sizeof(a), ""); // expected-error {{use of undeclared identifier 'a'}}
}

_Static_assert(1, "");

@end

struct S {
  @defs(A);
};

#else

// _Static_assert is available before C11 as an extension, but -pedantic
// warns on it.
#if __has_feature(objc_c_static_assert)
#error failed
#endif

#if !__has_extension(objc_c_static_assert)
#error failed
#endif

@interface A {
  int a;
  _Static_assert(1, "");
  _Static_assert(0, ""); // expected-error {{static_assert failed}}
}

_Static_assert(1, "");

@end

#endif
