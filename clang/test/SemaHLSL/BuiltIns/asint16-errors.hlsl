// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.2-library %s -fnative-half-type -verify


int16_t4 test_asint_too_many_arg(uint16_t p0, uint16_t p1)
{
    return asint16(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but 2 arguments were provided}}
}


int16_t test_asint_int(int p1)
{
    return asint16(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asint16<int>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<int, N>' against 'int'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = int16_t, T = int]: no type named 'Type'}}
}

int16_t test_asint_float(float p1)
{
    return asint16(p1);
    // expected-error@hlsl/hlsl_intrinsics.h:* {{no matching function for call to 'bit_cast'}}
    // expected-note@-2 {{in instantiation of function template specialization 'hlsl::asint16<float>'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: could not match 'vector<float, N>' against 'float'}}
    // expected-note@hlsl/hlsl_detail.h:* {{candidate template ignored: substitution failure [with U = int16_t, T = float]: no type named 'Type'}}
}
