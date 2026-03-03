// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library %s -verify

void no_arg() {
  __builtin_hlsl_interlocked_or();
  // expected-error@-1 {{too few arguments to function call, expected 3, have 0}}
}

void too_many_args() {
  __builtin_hlsl_interlocked_or(0, 0, 0, 0, 0);
  // expected-error@-1 {{too many arguments to function call, expected at most 4, have 5}}
}

void non_resource_arg() {
  __builtin_hlsl_interlocked_or(0, 0, 0);
  // expected-error@-1 {{used type 'int' where __hlsl_resource_t is required}}
}

void ret_no_arg() {
  __builtin_hlsl_interlocked_or_ret_uint();
  // expected-error@-1 {{too few arguments to function call, expected 4, have 0}}
}

void ret_too_many_args() {
  __builtin_hlsl_interlocked_or_ret_uint(0, 0, 0, 0, 0, 0);
  // expected-error@-1 {{too many arguments to function call, expected at most 5, have 6}}
}

void ret_non_resource_arg() {
  __builtin_hlsl_interlocked_or_ret_uint(0, 0, 0, 0);
  // expected-error@-1 {{used type 'int' where __hlsl_resource_t is required}}
}

// ByteAddressBuffer
using handle_char_t = __hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::raw_buffer]] [[hlsl::contained_type(char)]];
// Buffer<int>
using handle_int_t = __hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::contained_type(float)]];
// RWBuffer<float>
using handle_float_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float)]];

struct CustomResource {
  handle_char_t ByteAddressBufferChar;
  handle_int_t BufferFloat;
  handle_float_t RWBufferFloat;
};

void invalid_byte_address_buffer(CustomResource CR) {
  __builtin_hlsl_interlocked_or(CR.ByteAddressBufferChar, 0, 0);
  // expected-error@-1 {{invalid __hlsl_resource_t type attributes}}
}

void invalid_typed_buffer(CustomResource CR) {
  __builtin_hlsl_interlocked_or(CR.BufferFloat, 0u, 0u);
  // expected-error@-1 {{invalid __hlsl_resource_t type attributes}}
}

void invalid_rw_typed_buffer(CustomResource CR) {
  __builtin_hlsl_interlocked_or(CR.RWBufferFloat, 0u, 0);
  // expected-error@-1 {{invalid __hlsl_resource_t type attributes}}
}

void ret_invalid_byte_address_buffer(CustomResource CR) {
  __builtin_hlsl_interlocked_or_ret_uint(CR.ByteAddressBufferChar, 0u, 0, 0);
  // expected-error@-1 {{invalid __hlsl_resource_t type attributes}}
}

void ret_invalid_typed_buffer(CustomResource CR) {
  __builtin_hlsl_interlocked_or_ret_uint(CR.BufferFloat, 0u, 0, 0);
  // expected-error@-1 {{invalid __hlsl_resource_t type attributes}}
}

void ret_invalid_rw_typed_buffer(CustomResource CR) {
  __builtin_hlsl_interlocked_or_ret_uint(CR.RWBufferFloat, 0u, 0, 0);
  // expected-error@-1 {{invalid __hlsl_resource_t type attributes}}
}
