// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify


int test_select_first_arg_wrong_type(int1 p0, int t0, int f0) {
  return select(p0, t0, f0); // No diagnostic expected.
}

int1 test_select_bool_vals_diff_vecs(bool p0, int1 t0, int1 f0) {
  return select<int1>(p0, t0, f0); // No diagnostic expected.
}

int2 test_select_vector_vals_not_vecs(bool2 p0, int t0,
                                               int f0) {
  return select(p0, t0, f0); // No diagnostic expected.
}

int1 test_select_vector_vals_wrong_size(bool2 p0, int1 t0, int1 f0) {
  return select<int,1>(p0, t0, f0); // expected-warning{{implicit conversion truncates vector: 'bool2' (aka 'vector<bool, 2>') to 'vector<bool, 1>' (vector of 1 'bool' value)}}
}

int test_select_no_args() {
  return __builtin_hlsl_select(); // expected-error{{too few arguments to function call, expected 3, have 0}}
}

int test_select_builtin_wrong_arg_count(bool p0) {
  return __builtin_hlsl_select(p0); // expected-error{{too few arguments to function call, expected 3, have 1}}
}

// __builtin_hlsl_select tests
int test_select_builtin_wrong_arg_count2(bool p0, int t0) {
  return __builtin_hlsl_select(p0, t0); // expected-error{{too few arguments to function call, expected 3, have 2}}
}

int test_too_many_args(bool p0, int t0, int f0, int g0) {
  return __builtin_hlsl_select(p0, t0, f0, g0); // expected-error{{too many arguments to function call, expected 3, have 4}}
}

// not a bool or a vector of bool. should be 2 errors.
int test_select_builtin_first_arg_wrong_type1(int p0, int t0, int f0) {
  return __builtin_hlsl_select(p0, t0, f0); // expected-error{{invalid operand of type 'int' where 'bool' or a vector of such type is required}}
}

int test_select_builtin_first_arg_wrong_type2(int1 p0, int t0, int f0) {
  return __builtin_hlsl_select(p0, t0, f0); // expected-error{{invalid operand of type 'int1' (aka 'vector<int, 1>') where 'bool' or a vector of such type is required}}
}

// if a bool last 2 args are of same type
int test_select_builtin_bool_incompatible_args(bool p0, int t0, double f0) {
  return __builtin_hlsl_select(p0, t0, f0); // expected-error{{arguments are of different types ('int' vs 'double')}}
}

// if a vector second arg isnt a vector
int2 test_select_builtin_second_arg_not_vector(bool2 p0, int t0, int2 f0) {
  return __builtin_hlsl_select(p0, t0, f0); // No diagnostic expected.
}

// if a vector third arg isn't a vector
int2 test_select_builtin_second_arg_not_vector(bool2 p0, int2 t0, int f0) {
  return __builtin_hlsl_select(p0, t0, f0); // No diagnostic expected.
}

// if vector last 2 aren't same type (so both are vectors but wrong type)
int1 test_select_builtin_diff_types(bool1 p0, int1 t0, float1 f0) {
  return __builtin_hlsl_select(p0, t0, f0); // expected-error{{second and third arguments to __builtin_hlsl_select must be of scalar or vector type with matching scalar element type: 'vector<int, [...]>' vs 'vector<float, [...]>'}}
}
