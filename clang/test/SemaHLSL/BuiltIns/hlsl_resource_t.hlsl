// RUN: %clang_cc1 -fsyntax-only -verify -triple dxil-unknown-shadermodel6.3-library %s

// Note: As HLSL resource type are sizeless type, we don't exhaustively
// test for cases covered by sizeless-1.c and similar tests.

typedef int __hlsl_resource_t; // expected-error {{cannot combine with previous 'int' declaration specifier}} expected-warning {{typedef requires a name}}
typedef int __hlsl_resource_t[]; // expected-error {{cannot combine with previous 'int' declaration specifier}} expected-error {{expected unqualified-id}}

__hlsl_resource_t r1;
__hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
__hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
groupshared __hlsl_resource_t r11;
groupshared __hlsl_resource_t r12[10]; // expected-error {{array has sizeless element type 'groupshared __hlsl_resource_t'}}
groupshared __hlsl_resource_t r13[]; // expected-error {{array has sizeless element type 'groupshared __hlsl_resource_t'}}

static __hlsl_resource_t r21;
static __hlsl_resource_t r22[10]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
static __hlsl_resource_t r23[]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}

cbuffer CB {
  __hlsl_resource_t r31;
  __hlsl_resource_t r32[10]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
  __hlsl_resource_t r33[]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
}

struct S {
  __hlsl_resource_t r1;
  __hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
  __hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
};

class C {
  __hlsl_resource_t r1;
  __hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
  __hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
};

union U {
  __hlsl_resource_t r1;
  __hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
  __hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
};

void f1(__hlsl_resource_t r1);
void f2(__hlsl_resource_t r2[10]); // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
void f3(__hlsl_resource_t r3[]);    // expected-error {{array has sizeless element type '__hlsl_resource_t'}}

__hlsl_resource_t f4();

void f(__hlsl_resource_t arg) {
  __hlsl_resource_t r1;
  __hlsl_resource_t r2[10]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}
  __hlsl_resource_t r3[]; // expected-error {{array has sizeless element type '__hlsl_resource_t'}}

  static __hlsl_resource_t r4;

  __hlsl_resource_t foo = arg;
  int a = arg; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__hlsl_resource_t'}}
  int b = arg[0]; // expected-error {{subscripted value is not an array, pointer, or vector}}

  foo == arg; // expected-error {{invalid operands to binary expression ('__hlsl_resource_t' and '__hlsl_resource_t')}}
  foo + arg; // expected-error {{invalid operands to binary expression ('__hlsl_resource_t' and '__hlsl_resource_t')}}
  foo && arg; // expected-error {{invalid operands to binary expression ('__hlsl_resource_t' and '__hlsl_resource_t')}} expected-error {{value of type '__hlsl_resource_t' is not contextually convertible to 'bool'}}
  arg++; // expected-error {{cannot increment value of type '__hlsl_resource_t'}}
}
