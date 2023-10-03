// RUN: %clang_cc1 %s -fsyntax-only -verify

void __attribute__((annotate("foo"))) foo(float *a) {
  __attribute__((annotate("bar"))) int x;
  [[clang::annotate("bar")]] int x2;
  __attribute__((annotate(1))) int y; // expected-error {{expected string literal as argument of 'annotate' attribute}}
  [[clang::annotate(1)]] int y2; // expected-error {{expected string literal as argument of 'annotate' attribute}}
  __attribute__((annotate("bar", 1))) int z;
  [[clang::annotate("bar", 1)]] int z2;

  int u = __builtin_annotation(z, (char*) 0); // expected-error {{second argument to __builtin_annotation must be a non-wide string constant}}
  int v = __builtin_annotation(z, (char*) L"bar"); // expected-error {{second argument to __builtin_annotation must be a non-wide string constant}}
  int w = __builtin_annotation(z, "foo");
  float b = __builtin_annotation(*a, "foo"); // expected-error {{first argument to __builtin_annotation must be an integer}}

  __attribute__((annotate())) int c; // expected-error {{'annotate' attribute takes at least 1 argument}}
  [[clang::annotate()]] int c2;      // expected-error {{'annotate' attribute takes at least 1 argument}}
}
