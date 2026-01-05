// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify -DTEST_FUNC=__builtin_hlsl_or
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify -DTEST_FUNC=__builtin_hlsl_and


bool test_too_few_arg(bool a)
{
    return TEST_FUNC(a);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

bool test_too_many_arg(bool a)
{
    return TEST_FUNC(a, a, a);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

bool2 test_mismatched_args(bool2 a, bool3 b)
{
    return TEST_FUNC(a, b);
  // expected-error@-1 {{all arguments to}}{{_builtin_hlsl_or|_builtin_hlsl_and }}{{must have the same type}}
}

bool test_incorrect_type(int a)
{
    return TEST_FUNC(a, a);
  // expected-error@-1{{invalid operand of type 'int' where 'bool' or a vector or matrix of such type is required}}
}

bool test_mismatched_scalars(bool a, int b)
{
  return TEST_FUNC(a, b);
  // expected-error@-1{{all arguments to}}{{_builtin_hlsl_or|_builtin_hlsl_and }}{{must have the same type}}
}

bool2x2 test_mismatched_matrix_dimensions(bool2x2 a, bool3x3 b)
{
    return TEST_FUNC(a, b);
  // expected-error@-1{{cannot pass object of non-trivial type 'bool2x2' (aka 'matrix<bool, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@-2{{cannot pass object of non-trivial type 'bool3x3' (aka 'matrix<bool, 3, 3>') through variadic function; call will abort at runtime}}
  // expected-error@-3{{all arguments to}}{{_builtin_hlsl_or|_builtin_hlsl_and }}{{must have the same type}}
}
