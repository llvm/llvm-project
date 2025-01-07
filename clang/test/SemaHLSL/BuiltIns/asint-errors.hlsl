// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -verify


int4 test_asint_too_many_arg(float p0, float p1) {
  return asint(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asint'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but 2 arguments were provided}}
}

int test_asint_double(double p1) {
    return asint(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asint<double>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<double, N>' against 'double'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = int, T = double]: no type named 'Type'}}
}

int test_asint_half(half p1) {
    return asint(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asint<half>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<half, N>' against 'half'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = int, T = half]: no type named 'Type'}}
}
