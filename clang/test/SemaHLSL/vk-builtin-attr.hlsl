// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -verify %s

// No built-in names are supported yet, so every name is rejected. Only the
// paths that reject the attribute are exercised here.

// Invalid subjects.
// expected-error@+1 {{'vk::builtin' attribute only applies to parameters, non-static data members, and functions}}
[[vk::builtin("Bogus")]] static float global_var;

// expected-error@+1 {{'vk::builtin' attribute only applies to parameters, non-static data members, and functions}}
[[vk::builtin("Bogus")]] cbuffer InvalidCBuffer {
  float cb;
}

// Unsupported built-in name on each valid subject kind.
struct BadName {
  // expected-error-re@+1 {{'Bogus' is not a supported value for the 'vk::builtin' attribute; supported values are: {{.*}}}}
  [[vk::builtin("Bogus")]] float a : A;
  // The name is case sensitive.
  // expected-error-re@+1 {{'pointsize' is not a supported value for the 'vk::builtin' attribute; supported values are: {{.*}}}}
  [[vk::builtin("pointsize")]] float b : B;
  // expected-error-re@+1 {{'' is not a supported value for the 'vk::builtin' attribute; supported values are: {{.*}}}}
  [[vk::builtin("")]] float c : C;
};

// expected-error-re@+1 {{'Bogus' is not a supported value for the 'vk::builtin' attribute; supported values are: {{.*}}}}
float param([[vk::builtin("Bogus")]] float size : S) : A { return size; }

// expected-error-re@+1 {{'Bogus' is not a supported value for the 'vk::builtin' attribute; supported values are: {{.*}}}}
[[vk::builtin("Bogus")]] float func() : F { return 1.0f; }

// Wrong argument count and type.
struct BadArgs {
  // expected-error@+1 {{'vk::builtin' attribute takes one argument}}
  [[vk::builtin()]] float a : A;
  // expected-error@+1 {{'vk::builtin' attribute takes one argument}}
  [[vk::builtin("Bogus", "Bogus")]] float b : B;
  // expected-error@+1 {{expected string literal as argument of 'builtin' attribute}}
  [[vk::builtin(5)]] float c : C;
};

// Templates: the argument is a string literal, so it is validated when the
// template is declared, not when it is instantiated.
template <typename T> struct Templated {
  // expected-error-re@+1 {{'Bogus' is not a supported value for the 'vk::builtin' attribute; supported values are: {{.*}}}}
  [[vk::builtin("Bogus")]] T a : A;
};
Templated<float> TemplatedInstance;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {}
