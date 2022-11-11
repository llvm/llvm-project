// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64 -target-feature +bf16 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64 -target-feature -bf16 %s

__bf16 test_cast_from_float(float in) {
  return (__bf16)in; // expected-error {{cannot type-cast to __bf16}}
}

__bf16 test_cast_from_float_literal(void) {
  return (__bf16)1.0f; // expected-error {{cannot type-cast to __bf16}}
}

__bf16 test_cast_from_int(int in) {
  return (__bf16)in; // expected-error {{cannot type-cast to __bf16}}
}

__bf16 test_cast_from_int_literal(void) {
  return (__bf16)1; // expected-error {{cannot type-cast to __bf16}}
}

__bf16 test_cast_bfloat(__bf16 in) {
  return (__bf16)in; // this one should work
}

float test_cast_to_float(__bf16 in) {
  return (float)in; // expected-error {{cannot type-cast from __bf16}}
}

int test_cast_to_int(__bf16 in) {
  return (int)in; // expected-error {{cannot type-cast from __bf16}}
}

__bf16 test_implicit_from_float(float in) {
  return in; // expected-error {{returning 'float' from a function with incompatible result type '__bf16'}}
}

__bf16 test_implicit_from_float_literal(void) {
  return 1.0f; // expected-error {{returning 'float' from a function with incompatible result type '__bf16'}}
}

__bf16 test_implicit_from_int(int in) {
  return in; // expected-error {{returning 'int' from a function with incompatible result type '__bf16'}}
}

__bf16 test_implicit_from_int_literal(void) {
  return 1; // expected-error {{returning 'int' from a function with incompatible result type '__bf16'}}
}

__bf16 test_implicit_bfloat(__bf16 in) {
  return in; // this one should work
}

float test_implicit_to_float(__bf16 in) {
  return in; // expected-error {{returning '__bf16' from a function with incompatible result type 'float'}}
}

int test_implicit_to_int(__bf16 in) {
  return in; // expected-error {{returning '__bf16' from a function with incompatible result type 'int'}}
}

__bf16 test_cond(__bf16 a, __bf16 b, _Bool which) {
  // Conditional operator _should_ be supported, without nonsense
  // complaints like 'types __bf16 and __bf16 are not compatible'
  return which ? a : b;
}

__bf16 test_cond_float(__bf16 a, __bf16 b, _Bool which) {
  return which ? a : 1.0f; // expected-error {{incompatible operand types ('__bf16' and 'float')}}
}

__bf16 test_cond_int(__bf16 a, __bf16 b, _Bool which) {
  return which ? a : 1; // expected-error {{incompatible operand types ('__bf16' and 'int')}}
}
