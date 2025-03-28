// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -verify


void test_arg_missing() {
  __builtin_hlsl_elementwise_clip();
 // expected-error@-1 {{too few arguments to function call, expected 1, have 0}} 
}

void test_too_many_args(float p1, float p2) {
  __builtin_hlsl_elementwise_clip(p1, p2);
 // expected-error@-1 {{too many arguments to function call, expected 1, have 2}} 
}

void test_first_arg_type_mismatch(bool p) {
  __builtin_hlsl_elementwise_clip(p);
 // expected-error@-1 {{invalid operand of type 'bool' where 'float' or a vector of such type is required}} 
}

void test_first_arg_type_mismatch_3(half3 p) {
  __builtin_hlsl_elementwise_clip(p);
 // expected-error@-1 {{invalid operand of type 'half3' (aka 'vector<half, 3>') where 'float' or a vector of such type is required}} 
}

void test_first_arg_type_mismatch_3(double p) {
  __builtin_hlsl_elementwise_clip(p);
 // expected-error@-1 {{invalid operand of type 'double' where 'float' or a vector of such type is required}} 
}
