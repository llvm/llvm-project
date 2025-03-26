// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.2-library %s -fnative-half-type -verify

uint16_t test_asuint16_less_argument()
{
    return asuint16();
  // expected-error@-1 {{no matching function for call to 'asuint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but no arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but no arguments were provided}}

}

int16_t4 test_asuint16_too_many_arg(uint16_t p0, uint16_t p1)
{
    return asuint16(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asuint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'V', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'F', but 2 arguments were provided}}

}

int16_t test_asuint16_int(int p1)
{
    return asuint16(p1);
  // expected-error@-1 {{no matching function for call to 'asuint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match 'vector<T, N>' against 'int'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = int]: no type named 'Type'}}
}

int16_t test_asuint16_float(float p1)
{
    return asuint16(p1);
  // expected-error@-1 {{no matching function for call to 'asuint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match 'vector<T, N>' against 'float'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float]: no type named 'Type'}}
}

int16_t4 test_asuint16_vector_int(int4 p1)
{
    return asuint16(p1);
  // expected-error@-1 {{no matching function for call to 'asuint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = int, N = 4]: no type named 'Type'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = int4]: no type named 'Type'}}
}

int16_t4 test_asuint16_vector_float(float4 p1)
{
    return asuint16(p1);
  // expected-error@-1 {{no matching function for call to 'asuint16'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float, N = 4]: no type named 'Type'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float4]: no type named 'Type'}}
}

