
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify -DTEST_FUNC=or
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify -DTEST_FUNC=and

bool2x2 test_mismatched_args(bool2x2 a, bool3x3 b)
{
  return TEST_FUNC(a, b);
  // expected-warning@-1 {{implicit conversion truncates matrix: 'bool3x3' (aka 'matrix<bool, 3, 3>') to 'matrix<bool, 2, 2>'}}
}

bool3x3 test_mismatched_args(bool3x3 a, bool2x2 b)
{
  return TEST_FUNC(a, b);
  // expected-error@-1 {{cannot initialize return object of type 'bool3x3' (aka 'matrix<bool, 3, 3>') with an rvalue of type 'matrix<bool, 2, 2>'}}
}

bool2x2 test_mismatched_args2(bool3x3 a, bool2x2 b)
{
  return TEST_FUNC(a, b);
  // expected-warning@-1 {{implicit conversion truncates matrix: 'bool3x3' (aka 'matrix<bool, 3, 3>') to 'matrix<bool, 2, 2>'}}
}

bool3x3 test_mismatched_return_larger(bool2x2 a, bool2x2 b)
{
  return TEST_FUNC(a, b);
  // expected-error@-1 {{cannot initialize return object of type 'matrix<[...], 3, 3>' with an rvalue of type 'matrix<[...], 2, 2>'}}
}

bool2x2 test_mismatched_return_smaller(bool3x3 a, bool3x3 b)
{
  return TEST_FUNC(a, b);
  // expected-warning@-1 {{implicit conversion truncates matrix: 'bool3x3' (aka 'matrix<bool, 3, 3>') to 'matrix<bool, 2, 2>'}}
}
