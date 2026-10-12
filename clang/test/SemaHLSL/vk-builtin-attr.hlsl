// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -verify %s

// Valid subjects: field, parameter, function.
struct Valid {
  [[vk::builtin("PointSize")]] float size : PSIZE;
};

float param([[vk::builtin("PointSize")]] float size : PSIZE) : A { return size; }

[[vk::builtin("PointSize")]] float func() : PSIZE { return 1.0f; }

// Invalid subjects.
// expected-error@+1 {{'vk::builtin' attribute only applies to parameters, non-static data members, and functions}}
[[vk::builtin("PointSize")]] static float global_var;

// expected-error@+1 {{'vk::builtin' attribute only applies to parameters, non-static data members, and functions}}
[[vk::builtin("PointSize")]] cbuffer InvalidCBuffer {
  float cb;
}

// Unsupported built-in name.
struct BadName {
  // expected-error@+1 {{'Bogus' is not a supported value for the 'vk::builtin' attribute; supported values are: 'PointSize'}}
  [[vk::builtin("Bogus")]] float a : A;
  // The name is case sensitive.
  // expected-error@+1 {{'pointsize' is not a supported value for the 'vk::builtin' attribute; supported values are: 'PointSize'}}
  [[vk::builtin("pointsize")]] float b : B;
  // expected-error@+1 {{'' is not a supported value for the 'vk::builtin' attribute; supported values are: 'PointSize'}}
  [[vk::builtin("")]] float c : C;
};

// Wrong argument count and type.
struct BadArgs {
  // expected-error@+1 {{'vk::builtin' attribute takes one argument}}
  [[vk::builtin()]] float a : A;
  // expected-error@+1 {{'vk::builtin' attribute takes one argument}}
  [[vk::builtin("PointSize", "PointSize")]] float b : B;
  // expected-error@+1 {{expected string literal as argument of 'builtin' attribute}}
  [[vk::builtin(5)]] float c : C;
};

// Templates: the argument is a string literal, so it is validated when the
// template is declared, not when it is instantiated.
template <typename T> struct Templated {
  // expected-error@+1 {{'Bogus' is not a supported value for the 'vk::builtin' attribute; supported values are: 'PointSize'}}
  [[vk::builtin("Bogus")]] T a : A;
  [[vk::builtin("PointSize")]] T b : PSIZE;
};
Templated<float> TemplatedInstance;

// Combining `vk::builtin` and `vk::location` is rejected, whatever the order.
struct Combined {
  // expected-error@+2 {{'vk::location' and 'vk::builtin' attributes are not compatible}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[vk::builtin("PointSize")]] [[vk::location(1)]] float a : A;
  // expected-error@+2 {{'vk::builtin' and 'vk::location' attributes are not compatible}}
  // expected-note@+1 {{conflicting attribute is here}}
  [[vk::location(2)]] [[vk::builtin("PointSize")]] float b : B;
};

// expected-error@+2 {{'vk::location' and 'vk::builtin' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[vk::builtin("PointSize")]] [[vk::location(0)]] float func_combined() : A;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {}
