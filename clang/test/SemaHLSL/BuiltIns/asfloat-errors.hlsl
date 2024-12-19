// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -verify


float4 test_float_too_many_arg(float p0, float p1) {
  return asfloat(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asfloat'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but 2 arguments were provided}}
}


float test_float_double(double p1) {
    return asfloat(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asfloat<double>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<double, N>' against 'double'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = float, T = double]: no type named 'Type'}}
}

float test_float_half(half p1) {
    return asfloat(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asfloat<half>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<half, N>' against 'half'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = float, T = half]: no type named 'Type'}}
}


float test_float_half(bool p1) {
    return asfloat(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asfloat<bool>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<bool, N>' against 'bool'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = float, T = bool]: no type named 'Type'}}
}
