// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -verify %s

export vector<double, 3> test_parameter(vector<double, 5> vec5) {
  vec5.x = 1; // expected-error {{invalid swizzle 'x' on vector of over 4 elements}}
  return vec5.xyw; // expected-error {{invalid swizzle 'xyw' on vector of over 4 elements}}
}

export vector<double, 4> test_rgba(vector<double, 5> vec5) {
  return vec5.rgba; // expected-error {{invalid swizzle 'rgba' on vector of over 4 elements}}
}

