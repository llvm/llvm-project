// RUN: %clang_cc1 %s -fsyntax-only -verify

void __attribute__((annotate("foo"))) foo(float *a) { 
  __attribute__((annotate("bar"))) int x;
  __attribute__((annotate(1))) int y; // expected-error {{argument to annotate attribute was not a string literal}}
  __attribute__((annotate("bar", 1))) int z; // expected-error {{attribute takes one argument}}
  int u = __builtin_annotation(z, (char*) 0); // expected-error {{second argument to __builtin_annotation must be a non-wide string constant}}
  int v = __builtin_annotation(z, (char*) L"bar"); // expected-error {{second argument to __builtin_annotation must be a non-wide string constant}}
  int w = __builtin_annotation(z, "foo");
  float b = __builtin_annotation(*a, "foo"); // expected-error {{first argument to __builtin_annotation must be an integer}}
}
