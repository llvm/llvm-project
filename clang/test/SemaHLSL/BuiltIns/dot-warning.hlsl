// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm \
// RUN:   -disable-llvm-passes -Wimplicit-int-float-conversion-verify -verify-ignore-unexpected


float test_dot_builtin_vector_elem_size_reduction ( int64_t2 p0, float p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-Warning@-1 {{implicit conversion from 'int64_t2' (aka 'vector<int64_t, 2>') to '__attribute__((__vector_size__(2 * sizeof(float)))) float' (vector of 2 'float' values) may lose precision}}
}

float test_dot_builtin_int_vector_elem_size_reduction ( int2 p0, float p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-Warning@-1 {{implicit conversion from 'int2' (aka 'vector<int, 2>') to '__attribute__((__vector_size__(2 * sizeof(float)))) float' (vector of 2 'float' values) may lose precision}}
}
