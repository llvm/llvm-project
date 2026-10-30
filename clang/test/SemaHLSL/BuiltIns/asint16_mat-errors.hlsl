// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -verify


int16_t4x4 test_half_too_many_arg(half4x4 p0, half4x4 p1) {
  return asint16(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but 2 arguments were provided}}
}

int16_t4x4 test_double(double4x4 p1) {
  return asint16(p1);
  // expected-error@-1 {{no matching function for call to 'asint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = double, R = 4, C = 4]: no type named 'Type' in 'hlsl::__detail::enable_if<false, matrix<short, 4, 4>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match 'vector<T, N>' against 'double4x4' (aka 'matrix<double, 4, 4>')}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = double4x4]: no type named 'Type' in 'hlsl::__detail::enable_if<false, short>'}}
}

int16_t4x4 test_float(float4x4 p1) {
  return asint16(p1);
  // expected-error@-1 {{no matching function for call to 'asint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float, R = 4, C = 4]: no type named 'Type' in 'hlsl::__detail::enable_if<false, matrix<short, 4, 4>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match 'vector<T, N>' against 'float4x4' (aka 'matrix<float, 4, 4>')}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float4x4]: no type named 'Type' in 'hlsl::__detail::enable_if<false, short>'}}
}
