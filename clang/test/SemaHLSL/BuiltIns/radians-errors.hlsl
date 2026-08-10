// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

float test_too_few_arg() {
  return radians();
  // expected-error@-1 {{no matching function for call to 'radians'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires single argument 'Val', but no arguments were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* 20 {{candidate function not viable: requires single argument 'V', but no arguments were provided}}
}

float2 test_too_many_arg(float2 p0) {
  return radians(p0, p0);
  // expected-error@-1 {{no matching function for call to 'radians'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires single argument 'Val', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* 20 {{candidate function not viable: requires single argument 'V', but 2 arguments were provided}}
}

float test_bool_to_float_type_promotion(bool p1) {
  return radians(p1);
  // expected-error@-1 {{call to 'radians' is ambiguous}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* 3 {{candidate function}}
}

float1 test_vec1_inputs(float1 p0) {
  return radians(p0);
  // expected-warning@-1 {{implicit conversion turns vector to scalar: 'float1' (aka 'vector<float, 1>') to 'float'}}
}

typedef float float5 __attribute__((ext_vector_type(5)));

float5 test_vec5_inputs(float5 p0) {
  return radians(p0);
  // expected-error@-1 {{call to 'radians' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 4 {{candidate function}}
}
