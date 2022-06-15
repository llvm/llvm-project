// RUN: %clang_cc1 %s -fsyntax-only -fdouble-square-bracket-attributes -verify

const char *some_function();

void foo(float *[[clang::annotate_type("foo")]] a) {
  int [[clang::annotate_type("bar")]] x1;
  int *[[clang::annotate_type("bar")]] x2;
  int *[[clang::annotate_type("bar", 1)]] x3;
  int *[[clang::annotate_type("bar", 1 + 2)]] x4;
  struct {} [[clang::annotate_type("foo")]] x5;
  int [[clang::annotate_type("int")]] *[[clang::annotate_type("ptr")]] array[10] [[clang::annotate_type("arr")]];

  typedef int [[clang::annotate_type("bar")]] my_typedef;

  // GNU spelling is not supported
  int __attribute__((annotate_type("bar"))) y1;  // expected-warning {{unknown attribute 'annotate_type' ignored}}
  int *__attribute__((annotate_type("bar"))) y2; // expected-warning {{unknown attribute 'annotate_type' ignored}}

  // Various error cases
  // FIXME: We would want to prohibit the attribute on the following two lines.
  // However, Clang currently generally doesn't prohibit type-only C++11
  // attributes on declarations. This should be fixed more generally.
  [[clang::annotate_type("bar")]] int *z1;
  int *z2 [[clang::annotate_type("bar")]];
  [[clang::annotate_type("bar")]]; // expected-error {{'annotate_type' attribute cannot be applied to a statement}}
  int *[[clang::annotate_type(1)]] z3; // expected-error {{'annotate_type' attribute requires a string}}
  int *[[clang::annotate_type()]] z4; // expected-error {{'annotate_type' attribute takes at least 1 argument}}
  int *[[clang::annotate_type]] z5; // expected-error {{'annotate_type' attribute takes at least 1 argument}}
  int *[[clang::annotate_type(some_function())]] z6; // expected-error {{'annotate_type' attribute requires a string}}
  int *[[clang::annotate_type("bar", some_function())]] z7; // expected-error {{'annotate_type' attribute requires parameter 1 to be a constant expression}} expected-note{{subexpression not valid in a constant expression}}
  int *[[clang::annotate_type("bar", z7)]] z8; // expected-error {{'annotate_type' attribute requires parameter 1 to be a constant expression}} expected-note{{subexpression not valid in a constant expression}}
  int *[[clang::annotate_type("bar", int)]] z9; // expected-error {{expected expression}}
}
// More error cases: Prohibit adding the attribute to declarations.
// Different declarations hit different code paths, so they need separate tests.
// FIXME: Clang currently generally doesn't prohibit type-only C++11
// attributes on declarations.
[[clang::annotate_type("bar")]] int *global;
void annotated_function([[clang::annotate_type("bar")]] int);
void g([[clang::annotate_type("bar")]] int);
struct [[clang::annotate_type("foo")]] S;
struct [[clang::annotate_type("foo")]] S{
  [[clang::annotate_type("foo")]] int member;
  [[clang::annotate_type("foo")]] union {
    int i;
    float f;
  };
};
