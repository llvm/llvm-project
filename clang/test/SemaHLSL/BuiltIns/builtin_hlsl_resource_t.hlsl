// RUN: %clang_cc1 -fsyntax-only -verify -triple dxil-unknown-shadermodel6.3-library %s

// Note: As HLSL resource type are sizeless type, we don't exhaustively
// test for cases covered by sizeless-1.c and similar tests.

typedef int __builtin_hlsl_resource_t; // expected-error {{cannot combine with previous 'int' declaration specifier}} expected-warning {{typedef requires a name}}
typedef int __builtin_hlsl_resource_t[]; // expected-error {{cannot combine with previous 'int' declaration specifier}} expected-error {{expected unqualified-id}}

__builtin_hlsl_resource_t r1; // expected-error {{HLSL intangible type cannot be declared here}}
__builtin_hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
__builtin_hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
groupshared __builtin_hlsl_resource_t r11; // expected-error {{HLSL intangible type cannot be declared here}}
groupshared __builtin_hlsl_resource_t r12[10]; // expected-error {{array has sizeless element type 'groupshared __builtin_hlsl_resource_t'}}
groupshared __builtin_hlsl_resource_t r13[]; // expected-error {{array has sizeless element type 'groupshared __builtin_hlsl_resource_t'}}

static __builtin_hlsl_resource_t r21; // expected-error {{HLSL intangible type cannot be declared here}}
static __builtin_hlsl_resource_t r22[10]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
static __builtin_hlsl_resource_t r23[]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}

cbuffer CB {
  __builtin_hlsl_resource_t r31; // expected-error {{HLSL intangible type cannot be declared here}}
  __builtin_hlsl_resource_t r32[10]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r33[]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
}

struct S {
  __builtin_hlsl_resource_t r1; // expected-error {{field has sizeless type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
};

class C {
  __builtin_hlsl_resource_t r1; // expected-error {{field has sizeless type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
};

union U {
  __builtin_hlsl_resource_t r1; // expected-error {{field has sizeless type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
};

void f1(__builtin_hlsl_resource_t r1);     // expected-error {{HLSL intangible type cannot be used as function argument}}
void f2(__builtin_hlsl_resource_t r2[10]); // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
void f3(__builtin_hlsl_resource_t r3[]);    // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}

__builtin_hlsl_resource_t f4();     // expected-error {{HLSL intangible type cannot be used as function return value}}

void f(__builtin_hlsl_resource_t arg) { // expected-error {{HLSL intangible type cannot be used as function argument}}
  __builtin_hlsl_resource_t r1; // expected-error {{HLSL intangible type cannot be declared here}}
  __builtin_hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}
  __builtin_hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__builtin_hlsl_resource_t'}}

  static __builtin_hlsl_resource_t r4; // expected-error {{HLSL intangible type cannot be declared here}}

  __builtin_hlsl_resource_t foo = arg; // expected-error {{HLSL intangible type cannot be declared here}}
  int a = arg; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__builtin_hlsl_resource_t'}}
  int b = arg[0]; // expected-error {{subscripted value is not an array, pointer, or vector}}

  foo == arg; // expected-error {{invalid operands to binary expression ('__builtin_hlsl_resource_t' and '__builtin_hlsl_resource_t')}}
  foo + arg; // expected-error {{invalid operands to binary expression ('__builtin_hlsl_resource_t' and '__builtin_hlsl_resource_t')}}
  foo && arg; // expected-error {{invalid operands to binary expression ('__builtin_hlsl_resource_t' and '__builtin_hlsl_resource_t')}} expected-error {{value of type '__builtin_hlsl_resource_t' is not contextually convertible to 'bool'}}
  arg++; // expected-error {{cannot increment value of type '__builtin_hlsl_resource_t'}}
}
