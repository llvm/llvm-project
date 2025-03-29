// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -verify


uint4 test_asuint_too_many_arg(float p0, float p1) {
  return asuint(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asuint'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
}

uint test_asuint_double(double p1) {
    return asuint(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asuint<double>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<double, N>' against 'double'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = uint, T = double]: no type named 'Type'}}
}

uint test_asuint_half(half p1) {
    return asuint(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asuint<half>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<half, N>' against 'half'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = uint, T = half]: no type named 'Type'}}
}

void test_asuint_first_arg_const(double D) {
  const uint A = 0;
  uint B;
  asuint(D, A, B);
 // expected-error@hlsl/hlsl_intrinsics.h:* {{read-only variable is not assignable}} 
}

void test_asuint_second_arg_const(double D) {
  const uint A = 0;
  uint B;
  asuint(D, B, A);
 // expected-error@hlsl/hlsl_intrinsics.h:* {{read-only variable is not assignable}} 
}

void test_asuint_imidiate_value(double D) {
  uint B;
  asuint(D, B, 1);
 // expected-error@-1 {{cannot bind non-lvalue argument 1 to out paramemter}} 
}

void test_asuint_expr(double D) {
  uint B;
  asuint(D, B, B + 1);
 // expected-error@-1 {{cannot bind non-lvalue argument B + 1 to out paramemter}} 
}
