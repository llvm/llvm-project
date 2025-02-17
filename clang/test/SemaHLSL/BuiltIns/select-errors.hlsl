// RUN: %clang_cc1 -finclude-default-header
// -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only
// -disable-llvm-passes -verify -verify-ignore-unexpected

int test_no_arg() {
  return select();
  // expected-error@-1 {{no matching function for call to 'select'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template
  // not viable: requires 3 arguments, but 0 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not
  // viable: requires 3 arguments, but 0 were provided}}
}

int test_too_few_args(bool p0) {
  return select(p0);
  // expected-error@-1 {{no matching function for call to 'select'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not
  // viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not
  // viable: requires 3 arguments, but 1 was provided}}
}

int test_too_many_args(bool p0, int t0, int f0, int g0) {
  return select<int>(p0, t0, f0, g0);
  // expected-error@-1 {{no matching function for call to 'select'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not
  // viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not
  // viable: requires 3 arguments, but 4 were provided}}
}

int test_select_first_arg_wrong_type(int1 p0, int t0, int f0) {
  return select(p0, t0, f0);
  // expected-error@-1 {{no matching function for call to 'select'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not
  // viable: no known conversion from 'vector<int, 1>' (vector of 1 'int' value)
  // to 'bool' for 1st argument}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: could
  // not match 'vector<T, Sz>' against 'int'}}
}

int1 test_select_bool_vals_diff_vecs(bool p0, int1 t0, int1 f0) {
  return select<int1>(p0, t0, f0);
  // expected-warning@-1 {{implicit conversion truncates vector:
  // 'vector<int, 2>' (vector of 2 'int' values) to 'vector<int, 1>'
  // (vector of 1 'int' value)}}
}

int2 test_select_vector_vals_not_vecs(bool2 p0, int t0,
                                               int f0) {
  return select(p0, t0, f0);
  // expected-error@-1 {{no matching function for call to 'select'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored:
  // could not match 'vector<T, Sz>' against 'int'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not
  // viable: no known conversion from 'vector<bool, 2>'
  // (vector of 2 'bool' values) to 'bool' for 1st argument}}
}

int1 test_select_vector_vals_wrong_size(bool2 p0, int1 t0, int1 f0) {
  return select<int,1>(p0, t0, f0); // produce warnings
  // expected-warning@-1 {{implicit conversion truncates vector:
  // 'vector<bool, 2>' (vector of 2 'bool' values) to 'vector<bool, 1>'
  // (vector of 1 'bool' value)}}
  // expected-warning@-2 {{implicit conversion truncates vector:
  // 'vector<int, 2>' (vector of 2 'int' values) to 'vector<int, 1>'
  // (vector of 1 'int' value)}}
}

// __builtin_hlsl_select tests
int test_select_builtin_wrong_arg_count(bool p0, int t0) {
  return __builtin_hlsl_select(p0, t0);
  // expected-error@-1 {{too few arguments to function call, expected 3,
  // have 2}}
}

// not a bool or a vector of bool. should be 2 errors.
int test_select_builtin_first_arg_wrong_type1(int p0, int t0, int f0) {
  return __builtin_hlsl_select(p0, t0, f0);
  // expected-error@-1 {{passing 'int' to parameter of incompatible type
  // 'bool'}}
  // expected-error@-2 {{First argument to __builtin_hlsl_select must be of
  // vector type}}
  }

int test_select_builtin_first_arg_wrong_type2(int1 p0, int t0, int f0) {
  return __builtin_hlsl_select(p0, t0, f0);
  // expected-error@-1 {{passing 'vector<int, 1>' (vector of 1 'int' value) to
  // parameter of incompatible type 'bool'}}
  // expected-error@-2 {{First argument to __builtin_hlsl_select must be of
  // vector type}}
}

// if a bool last 2 args are of same type
int test_select_builtin_bool_incompatible_args(bool p0, int t0, double f0) {
  return __builtin_hlsl_select(p0, t0, f0);
  // expected-error@-1 {{arguments are of different types ('int' vs 'double')}}
}

// if a vector second arg isnt a vector
int2 test_select_builtin_second_arg_not_vector(bool2 p0, int t0, int2 f0) {
  return __builtin_hlsl_select(p0, t0, f0);
  // expected-error@-1 {{Second argument to __builtin_hlsl_select must be of
  // vector type}}
}

// if a vector third arg isn't a vector
int2 test_select_builtin_second_arg_not_vector(bool2 p0, int2 t0, int f0) {
  return __builtin_hlsl_select(p0, t0, f0);
  // expected-error@-1 {{Third argument to __builtin_hlsl_select must be of
  // vector type}}
}

// if vector last 2 aren't same type (so both are vectors but wrong type)
int2 test_select_builtin_diff_types(bool1 p0, int1 t0, float1 f0) {
  return __builtin_hlsl_select(p0, t0, f0);
  // expected-error@-1 {{arguments are of different types ('vector<int, [...]>'
  // vs 'vector<float, [...]>')}}
}
