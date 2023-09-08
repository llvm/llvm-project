// RUN: %clang_cc1 -fsyntax-only -Wnonnull -Wnullability %s -verify

#if __has_feature(nullability)
#else
#  error nullability feature should be defined
#endif


int * _Nullable foo(int * _Nonnull x);

int *_Nonnull ret_nonnull(void);

int *foo(int *x) {
  return 0;
}

int * _Nullable foo1(int * _Nonnull x); // expected-note {{previous declaration is here}}

int *foo1(int * _Nullable x) { // expected-warning {{nullability specifier '_Nullable' conflicts with existing specifier '_Nonnull'}}
  return 0;
}

int * _Nullable foo2(int * _Nonnull x);

int *foo2(int * _Nonnull x) {
  return 0;
}

int * _Nullable foo3(int * _Nullable x); // expected-note {{previous declaration is here}}

int *foo3(int * _Nonnull x) { // expected-warning {{nullability specifier '_Nonnull' conflicts with existing specifier '_Nullable'}}
  return 0;
}

int * ret_nonnull(void) {
  return 0; // expected-warning {{null returned from function that requires a non-null return value}}
}

int foo4(int * _Nonnull x, int * y) {
  return 0;
}

#define SAFE_CALL(X) if (X) foo(X)
int main (void) {
  foo(0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  (void)sizeof(foo(0)); // expect no diagnostic in unevaluated context.
  SAFE_CALL(0); // expect no diagnostic for unreachable code.
  foo4(
       0, // expected-warning {{null passed to a callee that requires a non-null argument}}
       0);
}
