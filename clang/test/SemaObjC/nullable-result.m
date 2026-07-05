// RUN: %clang_cc1                 -verify -fsyntax-only -fblocks -Wnullable-to-nonnull-conversion %s
// RUN: %clang_cc1 -xobjective-c++ -verify -fsyntax-only -fblocks -Wnullable-to-nonnull-conversion %s

@class X;
@class NSError;

_Static_assert(__has_feature(nullability_nullable_result), "");

@interface SomeClass
-(void)async_get:(void (^)(X *_Nullable_result rptr, NSError *err))completionHandler;
@end

void call(SomeClass *sc) {
  [sc async_get:^(X *_Nullable_result rptr, NSError *err) {}];
  [sc async_get:^(X *_Nullable rptr, NSError *err) {}];
}

void test_conversion(void) {
  X *_Nullable_result nr;
  X *_Nonnull l = nr; // expected-warning{{implicit conversion from nullable pointer 'X * _Nullable_result' to non-nullable pointer type 'X * _Nonnull'}}

  (void)^(X * _Nullable_result p) {
    X *_Nonnull l = p; // expected-warning{{implicit conversion from nullable pointer 'X * _Nullable_result' to non-nullable pointer type 'X * _Nonnull'}}
  };
}

void test_dup(void) {
  id _Nullable_result _Nullable_result a; // expected-warning {{duplicate nullability specifier '_Nullable_result'}}
  id _Nullable _Nullable_result b; // expected-error{{nullability specifier '_Nullable_result' conflicts with existing specifier '_Nullable'}}
  id _Nullable_result _Nonnull c; // expected-error{{nullability specifier '_Nonnull' conflicts with existing specifier '_Nullable_result'}}
}

@interface NoContextSensitive
-(nullable_result id)m; // expected-error {{expected a type}}
@property(assign, nullable_result) id p; // expected-error{{unknown property attribute 'nullable_result'}}
@end
