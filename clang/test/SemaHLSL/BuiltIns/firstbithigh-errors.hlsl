// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

int test_too_few_arg() {
  return firstbithigh();
  // expected-error@-1 {{no matching function for call to 'firstbithigh'}}
}

int test_too_many_arg(int p0) {
  return firstbithigh(p0, p0);
  // expected-error@-1 {{no matching function for call to 'firstbithigh'}}
}

double test_int_builtin(double p0) {
  return firstbithigh(p0);
  // expected-error@-1 {{call to 'firstbithigh' is ambiguous}}
}

double2 test_int_builtin_2(double2 p0) {
  return __builtin_hlsl_elementwise_firstbithigh(p0);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'double2' (aka 'vector<double, 2>'))}}
}

float test_int_builtin_3(float p0) {
  return __builtin_hlsl_elementwise_firstbithigh(p0);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'double')}}
}
