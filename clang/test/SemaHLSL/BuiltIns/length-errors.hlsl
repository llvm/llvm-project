// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

void test_too_few_arg()
{
  return length();
  // expected-error@-1 {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but no arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but no arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but no arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but no arguments were provided}}
}

void test_too_many_arg(float2 p0)
{
  return length(p0, p0);
  // expected-error@-1 {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but 2 arguments were provided}}
}

float double_to_float_type(double p0) {
  return length(p0);
  // expected-error@-1  {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = double]: no type named 'Type' in 'hlsl::__detail::enable_if<false, double>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = double]: no type named 'Type' in 'hlsl::__detail::enable_if<false, double>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match '__detail::HLSL_FIXED_VECTOR<half, N>' (aka 'vector<__detail::enable_if_t<(N > 1 && N <= 4), half>, N>') against 'double'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match '__detail::HLSL_FIXED_VECTOR<float, N>' (aka 'vector<__detail::enable_if_t<(N > 1 && N <= 4), float>, N>') against 'double'}}
}


float bool_to_float_type_promotion(bool p1)
{
  return length(p1);
  // expected-error@-1  {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{andidate template ignored: substitution failure [with T = bool]: no type named 'Type' in 'hlsl::__detail::enable_if<false, bool>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{andidate template ignored: substitution failure [with T = bool]: no type named 'Type' in 'hlsl::__detail::enable_if<false, bool>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match '__detail::HLSL_FIXED_VECTOR<half, N>' (aka 'vector<__detail::enable_if_t<(N > 1 && N <= 4), half>, N>') against 'bool'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match '__detail::HLSL_FIXED_VECTOR<float, N>' (aka 'vector<__detail::enable_if_t<(N > 1 && N <= 4), float>, N>') against 'bool'}}
}

float length_int_to_float_promotion(int p1)
{
  return length(p1);
  // expected-error@-1  {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = int]: no type named 'Type' in 'hlsl::__detail::enable_if<false, int>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = int]: no type named 'Type' in 'hlsl::__detail::enable_if<false, int>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match '__detail::HLSL_FIXED_VECTOR<half, N>' (aka 'vector<__detail::enable_if_t<(N > 1 && N <= 4), half>, N>') against 'int'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could not match '__detail::HLSL_FIXED_VECTOR<float, N>' (aka 'vector<__detail::enable_if_t<(N > 1 && N <= 4), float>, N>') against 'int'}}
}

float2 length_int2_to_float2_promotion(int2 p1)
{
  return length(p1);
  // expected-error@-1  {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = int2]}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = int2]}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{1st parameter does not match adjusted type 'vector<int, [...]>' of argument [with N = 2]}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{1st parameter does not match adjusted type 'vector<int, [...]>' of argument [with N = 2]}}
}

float1 test_vec1_inputs(float1 p0) {
  return length(p0);
  // expected-error@-1  {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 1>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 1>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, half>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, float>'}}
}

typedef float float5 __attribute__((ext_vector_type(5)));

float5 test_vec5_inputs(float5 p0) {
  return length(p0);
  // expected-error@-1  {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 5>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 5>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, half>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, float>'}}
}
