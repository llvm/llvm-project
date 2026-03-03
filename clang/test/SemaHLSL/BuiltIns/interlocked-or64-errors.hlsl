// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library %s -verify

using handle_long_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(long)]];

struct CustomResource {
  handle_long_t BufferLong;
};

void wrong_shader_model(CustomResource CR) {
  __builtin_hlsl_interlocked_or(CR.BufferLong, 0u, 0l);
  // expected-error@-1 {{intrinsic '__builtin_hlsl_interlocked_or(CR.BufferLong, 0U, 0L)' requires shader model 6.6 or greater}}
}

void ret_wrong_shader_model(CustomResource CR) {
  long ret;
  __builtin_hlsl_interlocked_or_ret_ll(CR.BufferLong, 0u, 0l, ret);
  // expected-error@-1 {{intrinsic '__builtin_hlsl_interlocked_or_ret_ll(CR.BufferLong, 0U, 0L, ret)' requires shader model 6.6 or greater}}
}
