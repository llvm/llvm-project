// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.8-library -verify %s

// vk::builtin only exists when targeting SPIR-V; elsewhere it is unknown.

struct S {
  // expected-warning@+1 {{unknown attribute 'vk::builtin' ignored}}
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

// expected-warning@+1 {{unknown attribute 'vk::builtin' ignored}}
float param([[vk::builtin("PointSize")]] float size : PSIZE) : A {
  return size;
}

// expected-warning@+1 {{unknown attribute 'vk::builtin' ignored}}
[[vk::builtin("PointSize")]] float func() : PSIZE { return 1.0f; }
