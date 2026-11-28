// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry main -fsyntax-only -verify %s

// A single decl can violate multiple PointSize rules at once -- combined
// with vk::location, wrong type, and wrong stage -- and all diagnostics
// fire together rather than short-circuiting on the first.
float main([[vk::location(5), vk::builtin("PointSize")]] int input : A) : B {
  // expected-error@-1 {{'vk::builtin' and 'vk::location' attributes are not compatible}}
  // expected-note@-2 {{conflicting attribute is here}}
  // expected-error@-3 {{'PointSize' builtin must be of type 'float'}}
  // expected-error@-4 {{'PointSize' builtin cannot be used as vertex shader inputs}}
  return (float)input;
}
