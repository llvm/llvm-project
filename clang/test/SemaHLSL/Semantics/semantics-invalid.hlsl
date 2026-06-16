// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -hlsl-entry main -verify %s

typedef float t_f : SEMANTIC; // expected-warning{{'SEMANTIC' attribute only applies to parameters, non-static data members, and functions}}

struct semantic_on_struct : SEMANTIC { // expected-error{{expected class name}}
  float a;
};

struct s_fields_multiple_semantics {
  float a : semantic_a : semantic_c; // expected-error{{use of undeclared identifier 'semantic_c'}}
  float b : semantic_b;
};

[numthreads(1, 1, 1)]
void main() {
  float a : SEM_A; // expected-warning{{'SEM_A' attribute only applies to parameters, non-static data members, and functions}}
}
