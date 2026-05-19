// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -verify


uint4x4 test_float_too_many_arg(float4x4 p0, float4x4 p1) {
  return asuint(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asuint'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
}


uint4x4 test_float_double(double4x4 p1) {
    return asuint(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asuint<double, 4, 4>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = uint, T = double, R = 4, C = 4]}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure: too many template arguments for function template 'bit_cast'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure: too many template arguments for function template 'bit_cast'}}
}

uint4x4 test_float_half(half4x4 p1) {
    return asuint(p1);
     // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asuint<half, 4, 4>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = uint, T = half, R = 4, C = 4]}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure: too many template arguments for function template 'bit_cast'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure: too many template arguments for function template 'bit_cast'}}
}
