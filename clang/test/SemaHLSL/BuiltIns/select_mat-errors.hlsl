// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

int2x2 test_first_arg_wrong_type(int2x2 p0, int2x2 t0, int2x2 f0) {
  // expected-error@+4{{cannot pass object of non-trivial type 'int2x2' (aka 'matrix<int, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+3{{cannot pass object of non-trivial type 'int2x2' (aka 'matrix<int, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+2{{cannot pass object of non-trivial type 'int2x2' (aka 'matrix<int, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+1{{invalid operand of type 'int2x2' (aka 'matrix<int, 2, 2>') where 'bool' or a vector or matrix of such type is required}}
  return __builtin_hlsl_select(p0, t0, f0);
}

int2x2 test_mismatched_dims(bool2x2 p0, int2x2 t0, int3x3 f0) {
  // expected-error@+4{{cannot pass object of non-trivial type 'bool2x2' (aka 'matrix<bool, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+3{{cannot pass object of non-trivial type 'int2x2' (aka 'matrix<int, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+2{{cannot pass object of non-trivial type 'int3x3' (aka 'matrix<int, 3, 3>') through variadic function; call will abort at runtime}}
  // expected-error@+1{{vector operands do not have the same number of elements ('bool2x2' (aka 'matrix<bool, 2, 2>') and 'int3x3' (aka 'matrix<int, 3, 3>'))}}
  return __builtin_hlsl_select(p0, t0, f0);
}

int2x2 test_mismatched_element_types(bool2x2 p0, int2x2 t0, float2x2 f0) {
  // expected-error@+4{{cannot pass object of non-trivial type 'bool2x2' (aka 'matrix<bool, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+3{{cannot pass object of non-trivial type 'int2x2' (aka 'matrix<int, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+2{{cannot pass object of non-trivial type 'float2x2' (aka 'matrix<float, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@+1{{second and third arguments to '__builtin_hlsl_select' must be of scalar or vector type with matching scalar element type: 'matrix<int, [2 * ...]>' vs 'matrix<float, [2 * ...]>'}}
  return __builtin_hlsl_select(p0, t0, f0);
}
