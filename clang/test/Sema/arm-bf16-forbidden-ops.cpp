// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64 -target-feature +bf16 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64 -target-feature -bf16 %s

__bf16 test_static_cast_from_float(float in) {
  return static_cast<__bf16>(in); // expected-error {{static_cast from 'float' to '__bf16' is not allowed}}
}

__bf16 test_static_cast_from_float_literal(void) {
  return static_cast<__bf16>(1.0f); // expected-error {{static_cast from 'float' to '__bf16' is not allowed}}
}

__bf16 test_static_cast_from_int(int in) {
  return static_cast<__bf16>(in); // expected-error {{static_cast from 'int' to '__bf16' is not allowed}}
}

__bf16 test_static_cast_from_int_literal(void) {
  return static_cast<__bf16>(1); // expected-error {{static_cast from 'int' to '__bf16' is not allowed}}
}

__bf16 test_static_cast_bfloat(__bf16 in) {
  return static_cast<__bf16>(in); // this one should work
}

float test_static_cast_to_float(__bf16 in) {
  return static_cast<float>(in); // expected-error {{static_cast from '__bf16' to 'float' is not allowed}}
}

int test_static_cast_to_int(__bf16 in) {
  return static_cast<int>(in); // expected-error {{static_cast from '__bf16' to 'int' is not allowed}}
}

__bf16 test_implicit_from_float(float in) {
  return in; // expected-error {{cannot initialize return object of type '__bf16' with an lvalue of type 'float'}}
}

__bf16 test_implicit_from_float_literal() {
  return 1.0f; // expected-error {{cannot initialize return object of type '__bf16' with an rvalue of type 'float'}}
}

__bf16 test_implicit_from_int(int in) {
  return in; // expected-error {{cannot initialize return object of type '__bf16' with an lvalue of type 'int'}}
}

__bf16 test_implicit_from_int_literal() {
  return 1; // expected-error {{cannot initialize return object of type '__bf16' with an rvalue of type 'int'}}
}

__bf16 test_implicit_bfloat(__bf16 in) {
  return in; // this one should work
}

float test_implicit_to_float(__bf16 in) {
  return in; // expected-error {{cannot initialize return object of type 'float' with an lvalue of type '__bf16'}}
}

int test_implicit_to_int(__bf16 in) {
  return in; // expected-error {{cannot initialize return object of type 'int' with an lvalue of type '__bf16'}}
}

__bf16 test_cond(__bf16 a, __bf16 b, bool which) {
  // Conditional operator _should_ be supported, without nonsense
  // complaints like 'types __bf16 and __bf16 are not compatible'
  return which ? a : b;
}

__bf16 test_cond_float(__bf16 a, __bf16 b, bool which) {
  return which ? a : 1.0f; // expected-error {{incompatible operand types ('__bf16' and 'float')}}
}

__bf16 test_cond_int(__bf16 a, __bf16 b, bool which) {
  return which ? a : 1; // expected-error {{incompatible operand types ('__bf16' and 'int')}}
}
