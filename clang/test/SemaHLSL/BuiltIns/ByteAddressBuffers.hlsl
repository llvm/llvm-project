// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library -x hlsl -fsyntax-only -verify %s

ByteAddressBuffer Buf : register(t0);
RWByteAddressBuffer RWBuf : register(u0);

uint test_load_uint_array()[4] {
  return Buf.Load<uint[4]>(0);
  // expected-error@-1 {{an array type is not allowed here}}
  // expected-note@-2 {{in instantiation of function template specialization 'hlsl::ByteAddressBuffer::Load<unsigned int[4]>' requested here}}
}

float test_load_float_array()[2] {
  return RWBuf.Load<float[2]>(0);
  // expected-error@-1 {{an array type is not allowed here}}
  // expected-note@-2 {{in instantiation of function template specialization 'hlsl::RWByteAddressBuffer::Load<float[2]>' requested here}}
}

uint test_load_uint_array_with_status()[4] {
  uint s1;
  return RWBuf.Load<uint[4]>(0, s1);
  // expected-error@-1 {{an array type is not allowed here}}
  // expected-note@-2 {{in instantiation of function template specialization 'hlsl::RWByteAddressBuffer::Load<unsigned int[4]>' requested here}}
}

float test_load_float_array_with_status()[2] {
  uint s1;
  return Buf.Load<float[2]>(0, s1);
  // expected-error@-1 {{an array type is not allowed here}}
  // expected-note@-2 {{in instantiation of function template specialization 'hlsl::ByteAddressBuffer::Load<float[2]>' requested here}}
}

void test_store_uint_array() {
  uint UIntArray[4];
  RWBuf.Store<uint[4]>(0, UIntArray);
  // expected-error@-1 {{an array type is not allowed here}}
  // expected-note@-2 {{in instantiation of function template specialization 'hlsl::RWByteAddressBuffer::Store<unsigned int[4]>' requested here}}
}

void test_store_float_array() {
  float FloatArray[2];
  RWBuf.Store<float[2]>(0, FloatArray);
  // expected-error@-1 {{an array type is not allowed here}}
  // expected-note@-2 {{in instantiation of function template specialization 'hlsl::RWByteAddressBuffer::Store<float[2]>' requested here}}
}
